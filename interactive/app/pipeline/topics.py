import hashlib
import sys
from collections import Counter

import numpy as np

import config
from config import ROLES

loaded = {}

def labeler():
    if "labels" not in loaded:
        if not config.EMBEDDINGS_DIR.exists():
            raise FileNotFoundError(f"embeddings code not found at {config.EMBEDDINGS_DIR} (set HEALTHPAL_EMBEDDINGS_DIR)")
        if str(config.EMBEDDINGS_DIR) not in sys.path:
            sys.path.insert(0, str(config.EMBEDDINGS_DIR))
        import llm_topic_labels
        loaded["labels"] = llm_topic_labels
    return loaded["labels"]

def embed_model():
    labeler()   # puts embeddings/ on sys.path
    from load_data import EMBED_MODEL   # the same model as the embeddings track, named in one place
    return EMBED_MODEL

def encoder():
    if "encoder" not in loaded:
        from sentence_transformers import SentenceTransformer
        loaded["encoder"] = SentenceTransformer(embed_model(), device="cpu")
    return loaded["encoder"]

def embed(labels):
    return encoder().encode(labels, normalize_embeddings=True, show_progress_bar=False)

# Cache key for a transcript frame based on its content (roles and text).
def cache_key(frame):
    content = "\n".join(f"{r}\t{t}" for r, t in zip(frame["role"], frame["text"]))
    return "transcript-" + hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]

# Get topic boundaries for a visit.
def boundaries(visit, df):
    llm = labeler()
    frame = df.rename(columns={"turn": "turn_index"})[["turn_index", "role", "text"]].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["word_count"] = frame["text"].str.split().str.len()
    cache_dir = config.PROCESSED_DIR / "topics"
    key = cache_key(frame)
    labels, window_of = llm.load_cache(cache_dir, key)
    if labels is None:   # labels shipped with the repo for the synthetic demo transcripts
        labels, window_of = llm.load_cache(config.TOPIC_SEED_DIR, key)
    if labels is None:
        try:
            labels, window_of = llm.compute_boundaries(frame)
        except Exception as error:
            raise RuntimeError(f"{error} (topic labeling needs Ollama running with {llm.MODEL}: "
                               f"`ollama serve`, then `ollama pull {llm.MODEL}`)") from error
        llm.save_cache(cache_dir, key, visit["meta"].get("filename", visit["csv"]), labels, window_of)
    return labels

# Cluster label embeddings; returns one group number per label
def group_labels(emb):
    if len(emb) < 2:
        return [0] * len(emb)
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=1 - config.TOPIC_SAME_COSINE).fit_predict(emb).tolist()

# The label most similar to the rest of its group, used as the group's name.
def central_label(labels, emb):
    if len(set(labels)) == 1:
        return labels[0]
    return labels[int(np.argmax(np.round((emb @ emb.T).sum(1), 4)))]   # ties -> earliest label

def run(visit, df, results):
    n = len(df)
    labels = boundaries(visit, df)
    # Turns before the LLM's first boundary are their own "opening" segment, as in
    # turn_dynamics.conversation_features (n_topics = boundaries + 1). Note that
    # llm_topic_labels.topic_duration_features folds them into the first topic instead; the
    # research features keep that definition so reported results don't change.
    starts = [0] + sorted(t for t in labels if 0 < t < n)
    ends = starts[1:] + [n]
    roles = df["role"].tolist()

    segments = [{"id": k, "start": s, "end": e - 1, "n_turns": e - s,
                 "label": labels.get(s) or "opening", "opened_by": roles[s]}
                for k, (s, e) in enumerate(zip(starts, ends))]
    seg_labels = [s["label"] for s in segments]
    emb = embed(seg_labels)   # encoded once; used for grouping and for naming each group
    group_of = group_labels(emb)

    threads = []
    for g in sorted(set(group_of), key=group_of.index):   # in order of first appearance
        segs = [s for s in segments if group_of[s["id"]] == g]
        # visits: runs of back-to-back segments; every run after the first is a return
        visits, returns = [[segs[0]]], []
        for prev, cur in zip(segs, segs[1:]):
            if cur["id"] == prev["id"] + 1:
                visits[-1].append(cur)
            else:
                visits.append([cur])
                returns.append(cur)
        threads.append({
            "id": len(threads),
            "label": central_label([s["label"] for s in segs], emb[[s["id"] for s in segs]]),
            "labels": list(dict.fromkeys(s["label"] for s in segs)),
            "segments": [s["id"] for s in segs],
            "visits": [{"start": v[0]["start"], "end": v[-1]["end"], "labels": [s["label"] for s in v]} for v in visits],
            "n_turns": sum(s["n_turns"] for s in segs),
            "repeated": len(visits) > 1,
            "returned_by": [s["opened_by"] for s in returns],
        })
        for s in segs:
            s["thread"] = threads[-1]["id"]

    turn_segment = [0] * n
    for s in segments:
        turn_segment[s["start"]:s["end"] + 1] = [s["id"]] * s["n_turns"]

    repeated = [t for t in threads if t["repeated"]]
    reopened = Counter(r for t in repeated for r in t["returned_by"])
    return {
        "llm_model": labeler().MODEL, "embed_model": embed_model(), "same_cosine": config.TOPIC_SAME_COSINE,
        "segments": segments, "threads": threads, "turn_segment": turn_segment,
        "summary": {
            "n_segments": len(segments), "n_topics": len(threads), "n_repeated": len(repeated),
            "n_returns": sum(len(t["visits"]) - 1 for t in repeated),
            "repeated_turn_share": round(100 * sum(t["n_turns"] for t in repeated) / n, 1) if n else 0.0,
            "returns_by_role": {r: reopened.get(r, 0) for r in ROLES},
        },
    }