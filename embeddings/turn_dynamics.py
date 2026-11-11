"""
Forrest Leung
11/2/26

This module implements turn-level topic dynamics for triadic dementia care visits.

To run:
python embeddings/turn_dynamics.py "$DATA"
"""
import argparse
import os

import numpy as np
import pandas as pd

import topic_segmentation as ts
from llm_topic_labels import DEFAULT_MERGE_COSINE_THRESHOLD, topic_duration_features
from load_data import EMBED_MODEL, FEATURE_ROLES, OTHER_ROLE
from load_data import get_turn_embeddings, key as visit_key
from load_data import load_dataset, load_roles, load_transcript

MIN_WORDS = 4   # shorter turns ("yeah", "mm-hm") are backchannels: kept for turn share, not "substantive"
ROLES = FEATURE_ROLES
DEFAULT_EMBEDDINGS_CACHE = "outputs/turn_embeddings"
DEFAULT_TOPICS_DIR = "outputs/topics"


# ---------- preprocessing ----------
def merge_turns(t):
    """Merge consecutive turns by the same speaker, since transcription often splits one
    turn across rows. A merged block counts as a topic boundary if any of the original
    turns inside it was flagged as one (new_topic column, from whichever source)."""
    block = (t["speaker"] != t["speaker"].shift()).cumsum()
    m = t.groupby(block, sort=False).agg(
        speaker=("speaker", "first"), role=("role", "first"),
        text=("text", " ".join),
        word_count=("word_count", "sum"),
        new_topic=("new_topic", "max"),
    ).reset_index(drop=True)
    m["substantive"] = m["word_count"] >= MIN_WORDS
    return m


# ---------- feature computation ----------
def conversation_features(m):
    n_turns = len(m)
    n_boundaries = int(m["new_topic"].sum())
    feats = {
        "n_turns": n_turns,
        "n_topics": n_boundaries + 1,   # turns before the first boundary are their own topic
        "topic_changes_per_turn": n_boundaries / max(n_turns, 1),
    }
    for r in ROLES:
        rm = m[m["role"] == r]
        sub = rm[rm["substantive"]]
        n_role_boundaries = int(rm["new_topic"].sum())
        feats[f"{r}_turn_share"] = len(rm) / max(n_turns, 1)           # 0 if absent, a true zero
        feats[f"{r}_mean_turn_length"] = float(rm["word_count"].mean()) if len(rm) else np.nan
        feats[f"{r}_topic_changes"] = n_role_boundaries                # 0 if absent, a true zero
        feats[f"{r}_topic_change_rate"] = n_role_boundaries / len(sub) if len(sub) else 0.0
    other_m = m[m["role"] == OTHER_ROLE]
    feats["other_turn_share"] = len(other_m) / max(n_turns, 1)   # 0 if absent, a true zero
    feats["has_other"] = int(len(other_m) > 0)
    feats["has_coordinator"] = int((m["role"] == "coordinator").any())

    # Patient turns that respond to a DIFFERENT speaker: do they introduce a new topic?
    prev_speaker = m["speaker"].shift()
    is_patient_response = (m["role"] == "patient") & prev_speaker.notna() & (prev_speaker != m["speaker"])
    resp = m[is_patient_response]
    feats["n_patient_responses"] = len(resp)
    feats["patient_topic_shift_rate"] = float(resp["new_topic"].mean()) if len(resp) else np.nan
    return feats


# ---------- boundary sources ----------
def boundaries_from_embeddings(t, path, roles, cache_dir, model, cutoff_sigma, context_k, min_gap):
    emb, _ = get_turn_embeddings(path, roles, cache_dir, model)
    return ts.embedding_boundaries(t, emb, MIN_WORDS, k=context_k,
                                   cutoff_sigma=cutoff_sigma, min_gap=min_gap)


def boundaries_from_llm(t, path, topics_dir):
    """Returns (new_topic array, raw labels dict, window_of dict) or (None, None, None)
    if uncached. window_of is {} for visits cached before provenance tracking existed."""
    from llm_topic_labels import load_cache
    labels, window_of = load_cache(topics_dir, visit_key(path))
    if labels is None:
        return None, None, None
    return t["turn_index"].isin(labels.keys()).astype(int).to_numpy(), labels, window_of


def merged_features(t, labels, window_of, model, cosine_threshold, min_segment_turns):
    """Same features as conversation_features()/topic_duration_features(), but from
    labels after merge_boundaries() -- prefixed merged_ so they never collide with
    the raw-boundary columns in the same output row."""
    from llm_topic_labels import merge_boundaries, topic_duration_features
    merged = merge_boundaries(labels, model, cosine_threshold, min_segment_turns, window_of=window_of,
                              first_turn=t["turn_index"].min(), last_turn=t["turn_index"].max())
    t2 = t.copy()
    t2["new_topic"] = t2["turn_index"].isin(merged.keys()).astype(int)
    m2 = merge_turns(t2)
    feats = conversation_features(m2)
    dur = topic_duration_features(t2, boundary_col="new_topic")
    dur.pop("n_topics", None)
    feats.update(dur)
    return {f"merged_{k}": v for k, v in feats.items()}


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="the 'Dementia Severity Data' folder")
    ap.add_argument("--boundary-source", choices=["llm", "embedding"], default="llm",
                    help="llm (default) = llm_topic_labels.py cache, the primary topic source; "
                         "embedding = topic_segmentation.py, comparison ablation only")
    ap.add_argument("--embeddings-cache", default=DEFAULT_EMBEDDINGS_CACHE,
                    help="--boundary-source embedding only: where turn embeddings are cached")
    ap.add_argument("--topics-dir", default=DEFAULT_TOPICS_DIR,
                    help="where llm_topic_labels.py cached boundaries (default: %(default)s)")
    ap.add_argument("--cutoff-sigma", type=float, default=ts.CUTOFF_SIGMA,
                    help="--boundary-source embedding only: threshold = mean(depth) + this * "
                         "std(depth); lower = more boundaries (default: %(default)s)")
    ap.add_argument("--context-k", type=int, default=ts.CONTEXT_K)
    ap.add_argument("--min-gap", type=int, default=ts.MIN_GAP_TURNS)
    ap.add_argument("--merge-cosine-threshold", type=float, default=DEFAULT_MERGE_COSINE_THRESHOLD,
                    help="--boundary-source llm only: merge adjacent segments whose topic "
                         "labels have cosine similarity >= this (default: %(default)s); "
                         "produces a second merged_* feature set, not a replacement")
    ap.add_argument("--min-segment-turns", type=int, default=None,
                    help="--boundary-source llm only: also fold any merged segment shorter "
                         "than this into its previous neighbor (default: off)")
    ap.add_argument("--skip-merge", action="store_true",
                    help="--boundary-source llm only: skip the merged_* feature set entirely")
    args = ap.parse_args()

    out_dir = os.path.join(args.data_dir, "derived")
    idx = load_dataset(args.data_dir, report_dir=out_dir)
    roles = load_roles(args.data_dir)

    model, merge_model = None, None
    if args.boundary_source == "embedding":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
    elif args.boundary_source == "llm" and not args.skip_merge:
        # CPU, not the default (MPS/Metal) device: confirmed MPS cosine similarities for
        # these short topic-label pairs are stable WITHIN a process but differ ACROSS
        # process runs (measured a 2-vs-57-drop swing on the same visit at the same
        # threshold, since many label pairs sit right at the merge boundary) -- CPU gave
        # bit-identical cosines across two separate runs. Merging ~20-100 short label
        # strings per visit is cheap enough that CPU costs nothing noticeable here.
        from sentence_transformers import SentenceTransformer
        merge_model = SentenceTransformer(EMBED_MODEL, device="cpu")

    rows, missing_topics, failed = [], [], []
    for i, row in idx.iterrows():
        vid = visit_key(row["path"])
        try:
            t = load_transcript(row["path"], roles)
        except Exception as e:   # one bad file shouldn't stop the run
            failed.append((os.path.relpath(row["path"], args.data_dir), f"{type(e).__name__}: {e}"))
            continue

        raw_labels, raw_window_of = None, None
        if args.boundary_source == "llm":
            new_topic, raw_labels, raw_window_of = boundaries_from_llm(t, row["path"], args.topics_dir)
            if new_topic is None:
                missing_topics.append(vid)
                continue
        else:
            new_topic = boundaries_from_embeddings(t, row["path"], roles, args.embeddings_cache, model,
                                                    args.cutoff_sigma, args.context_k, args.min_gap)
        t["new_topic"] = new_topic

        m = merge_turns(t)
        feats = conversation_features(m)
        dur_feats = topic_duration_features(t, boundary_col="new_topic")
        dur_feats.pop("n_topics", None)   # already in feats, from conversation_features
        feats.update(dur_feats)
        if raw_labels is not None and not args.skip_merge and len(raw_labels) >= 2:
            feats.update(merged_features(t, raw_labels, raw_window_of, merge_model, args.merge_cosine_threshold,
                                         args.min_segment_turns))
        feats.update({"visit_id": vid, "record_id_clean": row["record_id_clean"],
                      "fold": row["fold"], "site": row["site"], "timepoint": row["timepoint"],
                      "total_score": row["total_score"], "severity_category": row["severity_category"],
                      "boundary_source": args.boundary_source})
        rows.append(feats)
        print(f"\r  processed {i + 1}/{len(idx)} visits", end="", flush=True)
    print()

    if missing_topics:
        print(f"  skipped {len(missing_topics)}/{len(idx)} visits with no cached LLM topic labels; "
              f"run llm_topic_labels.py on them first (writing to {args.topics_dir})")
    if failed:
        print(f"  WARNING: skipped {len(failed)} transcripts that could not be read "
              f"(listed in derived/failed_transcripts.csv)")
        pd.DataFrame(failed, columns=["path", "error"]).to_csv(
            os.path.join(out_dir, "failed_transcripts.csv"), index=False)

    df = pd.DataFrame(rows)
    out = os.path.join(out_dir, "turn_dynamics_features.csv")
    df.to_csv(out, index=False)
    print(f"saved {len(df)} visits x {df.shape[1]} columns -> {out} "
          f"(boundary_source={args.boundary_source})")


if __name__ == "__main__":
    main()
