"""
Forrest Leung
10/26/26

Uses Ollama to pre-annotate topic boundaries for human review, cache the results one
file per visit, and compute topic-duration features from them.

To run:
pip install ollama pandas
ollama serve
python embeddings/llm_topic_labels.py "dir/example.csv"
python embeddings/llm_topic_labels.py "dir"/*.csv (whole folder)
python embeddings/llm_topic_labels.py "dir"/*.csv --features-out feats.csv (+ duration features on features)
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import ollama
import pandas as pd

from load_data import FEATURE_ROLES
from load_data import key as visit_key

MODEL = "gpt-oss:20b"
WINDOW = 40             # turns shown to the model per call
STRIDE = 30             # windows overlap by WINDOW - STRIDE turns
NUM_CTX = 8192          # context tokens; keep small on 16GB RAM
THINK = "low"           # gpt-oss reasoning effort; "low" is 6-11x faster than the default
                        # with no change in boundary positions (see label_window)
MAX_OUTPUT_TOKENS = 1024   # a safety cap, not the normal ceiling (~200 tokens/window at
                           # think="low"); without it a stuck/verbose generation can run away
MIN_WORDS_FOR_BOUNDARY = 4
DEFAULT_OUTPUTS_DIR = "outputs/topics"

# Version of the role scheme used in the prompt. Bump this whenever the role set or strings change.
ROLE_SCHEME_VERSION = "patient|caregiver|provider|coordinator|other-v2"

PROMPT = """You are annotating a transcript of a medical visit with a patient, their caregiver, and a provider.
Mark every turn that INTRODUCES a new topic. A topic is something a clinician would note as a separate
item (e.g. sleep, a specific medication, a fall, driving, a family event, greeting, wrap-up).

Rules:
- Mark the turn that first raises the new subject, not later turns that continue it.
- A change of speaker is NOT a topic change. Answers and follow-up questions stay in the same topic.
- Short backchannels ("yeah", "mm-hm", "okay") never start a topic.
- Returning to a subject discussed earlier DOES count as a new boundary.
- Moving to a subtopic of the same subject ("sleep" -> "naps") does NOT count.
- Turns before the window's first turn are not shown; do not mark the first turn unless it clearly opens a new subject.

Return JSON only: {{"boundaries": [{{"turn_index": <int>, "topic": "<2-4 words>"}}]}}

Transcript:
{turns}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "boundaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"turn_index": {"type": "integer"}, "topic": {"type": "string"}},
                "required": ["turn_index", "topic"],
            },
        }
    },
    "required": ["boundaries"],
}

# Loading
def load_turns(path):
    df = pd.read_csv(path, sep=None, engine="python")
    if "word_count" not in df:
        df["word_count"] = df["text"].astype(str).str.split().str.len()
    return df.sort_values("turn_index").reset_index(drop=True)


# Frozen labeling config
def labeling_config():
    return {
        "model": MODEL, "window": WINDOW, "stride": STRIDE, "num_ctx": NUM_CTX,
        "think": THINK, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "min_words_for_boundary": MIN_WORDS_FOR_BOUNDARY, "prompt": PROMPT,
        "role_scheme": ROLE_SCHEME_VERSION,
    }

def labeling_config_hash():
    blob = json.dumps(labeling_config(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]

# Caching
def cache_path(outputs_dir, visit_id):
    return os.path.join(outputs_dir, f"{visit_id}.json")

# Load cached labels for a given visit, returning (None, None) if not cached or stale.
def load_cache(outputs_dir, visit_id):
    p = cache_path(outputs_dir, visit_id)
    if not os.path.exists(p):
        return None, None
    with open(p) as f:
        data = json.load(f)
    if data.get("config_hash") != labeling_config_hash():
        return None, None
    labels = {int(k): v for k, v in data["labels"].items()}
    window_of = {int(k): v for k, v in data.get("window_of", {}).items()}
    return labels, window_of

def save_cache(outputs_dir, visit_id, src_path, labels, window_of=None):
    os.makedirs(outputs_dir, exist_ok=True)
    with open(cache_path(outputs_dir, visit_id), "w") as f:
        json.dump({"visit_id": visit_id, "source_name": Path(str(src_path)).name,
                   "model": MODEL, "config_hash": labeling_config_hash(),
                   "labels": {str(k): v for k, v in labels.items()},
                   "window_of": {str(k): v for k, v in (window_of or {}).items()}}, f, indent=2)

# Labeling
def label_window(chunk):
    turns = "\n".join(f"[{r.turn_index}] {str(r.role).upper()}: {r.text}" for r in chunk.itertuples())
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(turns=turns)}],
        format=SCHEMA,
        think=THINK,
        options={"temperature": 0, "num_ctx": NUM_CTX, "seed": 0, "num_predict": MAX_OUTPUT_TOKENS},
    )
    try:
        return json.loads(resp["message"]["content"])["boundaries"]
    except (json.JSONDecodeError, KeyError):
        print(f"  could not parse model output for turns {chunk.turn_index.iloc[0]}-"
              f"{chunk.turn_index.iloc[-1]}; skipping window")
        return []

# Compute topic boundaries for a given DataFrame of turns.
def compute_boundaries(df):
    valid = set(df["turn_index"])
    short = set(df.loc[df["word_count"] < MIN_WORDS_FOR_BOUNDARY, "turn_index"])
    labels, window_of = {}, {}
    starts = range(0, max(len(df) - WINDOW + STRIDE, 1), STRIDE)
    for n, start in enumerate(starts, 1):
        chunk = df.iloc[start:start + WINDOW]
        if chunk.empty:
            break
        print(f"  window {n}/{len(starts)}: turns {chunk.turn_index.iloc[0]}-{chunk.turn_index.iloc[-1]}")
        for b in label_window(chunk):
            t = b.get("turn_index")
            # Drop hallucinated indices and backchannels; keep the first label a turn gets
            if t in valid and t not in short and t not in labels:
                labels[t] = b.get("topic", "")
                window_of[t] = start
    return labels, window_of

def apply_boundaries(df, labels):
    out = df.copy()
    out["new_topic_llm"] = out["turn_index"].isin(labels).astype(int)
    out["topic_label_llm"] = out["turn_index"].map(labels).fillna("")
    out["new_topic"] = ""        # for your hand corrections
    out["topic_label"] = ""
    out["repeat_of"] = ""
    return out

# Post-processing merge (no LLM rerun)
STITCHING_GAP_MAX = WINDOW - STRIDE
DEFAULT_MERGE_COSINE_THRESHOLD = 0.75

# Merge LLM-labeled topic boundaries based on semantic similarity.
def merge_boundaries(labels, model, cosine_threshold=DEFAULT_MERGE_COSINE_THRESHOLD,
                     min_segment_turns=None, window_of=None, first_turn=None, last_turn=None,
                     dedup_stitching=False):
    idxs = sorted(labels.keys())
    if len(idxs) < 2:
        return dict(labels)
    embs = model.encode([labels[i] for i in idxs], normalize_embeddings=True)
    keep = [True] * len(idxs)

    if window_of and dedup_stitching:
        for i in range(len(idxs)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(idxs)):
                if idxs[j] - idxs[i] > STITCHING_GAP_MAX:
                    break
                if not keep[j]:
                    continue
                wi, wj = window_of.get(idxs[i]), window_of.get(idxs[j])
                if wi is None or wj is None or wi == wj:
                    continue   # same window, or provenance missing for one of the pair
                sim = float(embs[i] @ embs[j])
                if sim >= cosine_threshold:
                    drop_i = i if wi > wj else j   # drop the LATER window's proposal
                    keep[drop_i] = False
                    print(f"  stitching duplicate: turns {idxs[i]} & {idxs[j]} "
                         f"(windows {wi}, {wj}, cosine={sim:.2f}) -- dropped turn {idxs[drop_i]}")
                    if drop_i == i:
                        break

    anchor = next((i for i in range(len(idxs)) if keep[i]), None)
    if anchor is not None:
        for i in range(anchor + 1, len(idxs)):
            if not keep[i]:
                continue
            if float(embs[anchor] @ embs[i]) >= cosine_threshold:
                keep[i] = False
            else:
                anchor = i

    if min_segment_turns and first_turn is not None and last_turn is not None:
        changed = True
        while changed:
            changed = False
            surviving = [idxs[i] for i in range(len(idxs)) if keep[i]]
            starts = [first_turn] + surviving
            ends = surviving + [last_turn + 1]
            for seg_i, (s, e) in enumerate(zip(starts, ends)):
                if e - s < min_segment_turns and seg_i > 0:
                    pos = idxs.index(surviving[seg_i - 1])
                    keep[pos] = False
                    changed = True
                    break

    return {idxs[i]: labels[idxs[i]] for i in range(len(idxs)) if keep[i]}

# Topic duration features from topic boundaries
def topic_duration_features(df, boundary_col="new_topic_llm"):
    boundary = df[boundary_col].fillna(0).astype(int)
    seg = boundary.cumsum()
    # Turns before the first boundary belong to run 1. turn_dynamics.conversation_features and the
    # demo app's Topics step count them as a separate topic instead; this definition is kept as is
    # because the reported topic-duration results were computed with it.
    seg = seg.where(seg > 0, 1)

    turns_per_run = seg.value_counts()
    words_per_run = df.groupby(seg)["word_count"].sum()
    feats = {
        "n_topics": int(seg.nunique()),
        "topic_run_turns_mean": float(turns_per_run.mean()),
        "topic_run_turns_median": float(turns_per_run.median()),
        "topic_run_turns_max": float(turns_per_run.max()),
        "topic_run_words_mean": float(words_per_run.mean()),
        "topic_run_words_median": float(words_per_run.median()),
        "topic_run_words_max": float(words_per_run.max()),
    }
    if "role" in df.columns:
        runs = seg.unique()
        for role in FEATURE_ROLES:   # fixed set, so every visit has the same columns
            m = (df["role"] == role).to_numpy()
            if not m.any():          # role never speaks in this visit: a true NaN, not a 0
                feats[f"{role}_topic_turns_mean"] = float("nan")
                feats[f"{role}_topic_words_mean"] = float("nan")
                continue
            t_counts = seg[m].value_counts().reindex(runs, fill_value=0)
            w_counts = df.loc[m].groupby(seg[m])["word_count"].sum().reindex(runs, fill_value=0)
            feats[f"{role}_topic_turns_mean"] = float(t_counts.mean())
            feats[f"{role}_topic_words_mean"] = float(w_counts.mean())
    return feats


# Process a single transcript: load turns, get topic boundaries, apply them, and return the labeled DataFrame.
def process_one(path, outputs_dir):
    p = Path(path)
    visit_id = visit_key(p)
    df = load_turns(p)
    if df.empty:
        print(f"  {p.name}: no turns parsed; skipping")
        return None, None
    labels, window_of = load_cache(outputs_dir, visit_id)
    if labels is not None:
        print(f"{p.name}: using cached labels ({len(labels)} boundaries)")
    else:
        print(p.name)
        labels, window_of = compute_boundaries(df)
        save_cache(outputs_dir, visit_id, p, labels, window_of)
        print(f"  {len(labels)} boundaries in {len(df)} turns -> {cache_path(outputs_dir, visit_id)}")
    return visit_id, apply_boundaries(df, labels)

def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--outputs-dir", default=DEFAULT_OUTPUTS_DIR,
                    help="where per-visit topic-boundary JSON is cached/read (default: %(default)s)")
    ap.add_argument("--write-csv", action="store_true",
                    help="also write <name>_llm_labels.csv next to each source file, for hand review")
    ap.add_argument("--features-out", help="write one row per visit of topic-duration features to this CSV")
    args = ap.parse_args(argv)
    print(f"labeling config hash: {labeling_config_hash()}  (model={MODEL}, role_scheme={ROLE_SCHEME_VERSION})")

    feature_rows = []
    for raw in args.csvs:
        p = Path(raw)
        if p.suffix.lower() != ".csv" or p.stem.endswith("_llm_labels"):
            continue  # skip non-CSVs and this script's own CSV output on reruns
        visit_id, out = process_one(p, args.outputs_dir)
        if out is None:
            continue
        if args.write_csv:
            dest = p.with_name(p.stem + "_llm_labels.csv")
            out.to_csv(dest, index=False)
        if args.features_out:
            feature_rows.append({"visit_id": visit_id, "path": str(p), **topic_duration_features(out)})

    if args.features_out:
        pd.DataFrame(feature_rows).to_csv(args.features_out, index=False)
        print(f"wrote topic-duration features for {len(feature_rows)} visits -> {args.features_out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])