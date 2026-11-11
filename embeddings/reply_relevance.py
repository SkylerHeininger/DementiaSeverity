"""
Forrest Leung
11/11/26

Compute reply relevance features for each visit based on the embeddings of provider and patient/caregiver turns.

To run (once per embedding model):
python embeddings/reply_relevance.py "$DATA"
python embeddings/reply_relevance.py "$DATA" --model sentence-transformers/all-mpnet-base-v2
python embeddings/reply_relevance.py "$DATA" --stats
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as student_t

from load_data import EMBED_MODEL, clean_col, get_turn_embeddings, key as visit_key
from load_data import baseline_visits, load_dataset, load_roles, load_transcript

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DASEG_DIR = os.path.join(HERE, "..", "predict_severity", "VCs1_All_Transcriptions_daseg")
DASEG_SUFFIX = "_with_predictions"

MIN_WORDS = 4
MIN_REPLIES = 3
MIN_POOL = 2
RESPONDER_ROLES = ("patient", "caregiver")
SUBSETS = ("all", "q")

QUESTION_DAS = {
    "Wh-Question", "Yes-No-Question", "Open-Question", "Or-Clause",
    "Declarative-Yes-No-Question", "Rhetorical-Questions",
    "Tag-Question", "Declarative-Wh-Question",
}

def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

def model_slug(model_name):
    return re.sub(r"[^a-z0-9]+", "_", model_name.split("/")[-1].lower()).strip("_")

# Resolve the cache directory for a given model, ensuring it matches the expected model.
def resolve_cache_dir(model_name, cache_dir=None):
    if cache_dir is None:
        name = "turn_embeddings" if model_name == EMBED_MODEL else f"turn_embeddings_{model_slug(model_name)}"
        cache_dir = os.path.join(HERE, "outputs", name)
    sidecar = os.path.join(cache_dir, "MODEL.txt")
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            cached = f.read().strip()
        if cached != model_name:
            raise SystemExit(f"{cache_dir} holds embeddings from {cached!r}, not {model_name!r}; "
                             f"pass a different --embeddings-cache")
        return cache_dir
    has_npz = bool(glob.glob(os.path.join(cache_dir, "*.npz")))
    if has_npz and model_name != EMBED_MODEL:
        raise SystemExit(f"{cache_dir} already has embeddings from an unrecorded model (no MODEL.txt); "
                         f"pass an empty --embeddings-cache for {model_name!r}")
    os.makedirs(cache_dir, exist_ok=True)
    with open(sidecar, "w") as f:
        f.write(model_name + "\n")
    return cache_dir

# Index DASeg files by visit key.
def daseg_index(daseg_dir):
    out = {}
    for p in glob.glob(os.path.join(daseg_dir, f"*{DASEG_SUFFIX}.csv")):
        stem = os.path.splitext(os.path.basename(p))[0]
        out[visit_key(stem[: -len(DASEG_SUFFIX)])] = p
    return out

# Identify turn indices whose last dialogue act is a question.
def question_turns(daseg_csv):
    try:
        d = pd.read_csv(daseg_csv, dtype=str, keep_default_na=False)
    except Exception:
        return None
    d.columns = [clean_col(c) for c in d.columns]
    if not {"turn_index", "proc_da"} <= set(d.columns):
        return None
    d["da"] = d["proc_da"].str.strip().str.replace(r"^[BI]-", "", regex=True)
    d["turn_index"] = pd.to_numeric(d["turn_index"], errors="coerce")
    d = d[(d["da"] != "") & d["turn_index"].notna()]
    last_da = d.groupby("turn_index", sort=False)["da"].last()
    return {int(t) for t in last_da.index[last_da.isin(QUESTION_DAS)]}

# Merge consecutive rows by speaker into weighted blocks.
def merge_blocks(t, emb):
    block_id = (t["speaker"] != t["speaker"].shift()).cumsum().to_numpy()
    words = t["word_count"].to_numpy(float)
    blocks = []
    for b in pd.unique(block_id):
        rows = np.where(block_id == b)[0]
        w = np.clip(words[rows], 0, None)
        v = (emb[rows] * w[:, None]).sum(0) if w.sum() > 0 else emb[rows].mean(0)
        blocks.append({"role": t["role"].iat[rows[0]], "words": float(w.sum()),
                       "last_turn": int(t["turn_index"].iat[rows[-1]]), "emb": unit(v)})
    return blocks

# Build reply records for provider prompts followed by responder blocks.
def reply_records(blocks, questions=None):
    def ends_in_question(b):
        return questions is not None and b["last_turn"] in questions

    prompts = [i for i, b in enumerate(blocks)
               if b["role"] == "provider" and (b["words"] >= MIN_WORDS or ends_in_question(b))]
    if not prompts:
        return []
    P = np.stack([blocks[i]["emb"] for i in prompts])
    pos = {i: k for k, i in enumerate(prompts)}
    recs = []
    for i in prompts:
        j = i + 1
        if j >= len(blocks):
            continue
        r = blocks[j]
        if r["role"] not in RESPONDER_ROLES or r["words"] < MIN_WORDS:
            continue
        sims = P @ r["emb"]
        others = np.delete(sims, pos[i])
        recs.append({
            "role": r["role"],
            "raw": float(sims[pos[i]]),
            "excess": float(sims[pos[i]] - others.mean()) if len(others) >= MIN_POOL else np.nan,
            "is_q": ends_in_question(blocks[i]),
        })
    return recs

# Summarize reply features by responder role and question subset.
def visit_features(recs, have_questions):
    rec = pd.DataFrame(recs, columns=["role", "raw", "excess", "is_q"])
    feats = {}
    for subset in SUBSETS:
        sub = rec if subset == "all" else rec[rec["is_q"].astype(bool)]
        for role in RESPONDER_ROLES:
            r = sub[sub["role"] == role].dropna(subset=["excess"])
            usable = len(r) >= MIN_REPLIES and (subset == "all" or have_questions)
            feats[f"n_reply_{role}_{subset}"] = len(r) if (subset == "all" or have_questions) else np.nan
            feats[f"reply_{role}_raw_{subset}"] = float(r["raw"].mean()) if usable else np.nan
            feats[f"reply_{role}_excess_{subset}"] = float(r["excess"].mean()) if usable else np.nan
            feats[f"reply_{role}_offtarget_{subset}"] = float((r["excess"] <= 0).mean()) if usable else np.nan
        feats[f"reply_patient_minus_caregiver_excess_{subset}"] = (
            feats[f"reply_patient_excess_{subset}"] - feats[f"reply_caregiver_excess_{subset}"])
    return feats

# Report Spearman correlation and p-value, or a note when too few observations are available for H3.
def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 5:
        return f"n={ok.sum()} (too few)"
    r = spearmanr(a[ok], b[ok])
    return f"rho={r.correlation:+.3f} p={r.pvalue:.3g} n={ok.sum()}"

# Report Spearman partial correlation and p-value, or a note when too few observations are available for H3.
def partial_rho(x, y, Z):
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.asarray(Z, float).reshape(len(x), -1)
    ok = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(Z).any(axis=1)
    n, k = int(ok.sum()), Z.shape[1]
    if n - k < 6:
        return f"n={n} (too few)"
    R = np.column_stack([np.ones(n)] + [rankdata(Z[ok, j]) for j in range(k)])
    def resid(v):
        return v - R @ np.linalg.lstsq(R, v, rcond=None)[0]
    r = float(np.corrcoef(resid(rankdata(x[ok])), resid(rankdata(y[ok])))[0, 1])
    dof = n - 2 - k
    p = 2 * student_t.sf(abs(r) * np.sqrt(dof / max(1 - r ** 2, 1e-12)), dof)
    return f"rho={r:+.3f} p={p:.3g} n={n}"

# The handful of features the checks focus on
KEY_FEATURES = [f"reply_{who}_excess_{s}" for s in SUBSETS for who in ("patient", "caregiver")] + \
               [f"reply_patient_minus_caregiver_excess_{s}" for s in SUBSETS]
TALK_CONTROLS = [f"ctl_share_{r}" for r in RESPONDER_ROLES]

def run_stats(idx, df):
    d = idx[["visit_id", "record_id_clean", "site", "timepoint", "total_score"]].merge(df, on="visit_id")
    first = baseline_visits(d)   # earliest timepoint per patient: one row per patient
    print(f"\n{len(d)} visits, {d['record_id_clean'].nunique()} patients "
          f"({d['daseg_available'].sum()} visits with DASeg questions)")

    print("\nmeasure validity (no DSRS involved): mean excess should be > 0, i.e. replies are")
    print("closer to the prompt they answer than to the visit's other provider prompts")
    for subset in SUBSETS:
        for role in RESPONDER_ROLES:
            col = f"reply_{role}_excess_{subset}"
            v = d[col].dropna()
            print(f"  {col:40s} mean={v.mean():+.4f}  visits={len(v)}")

    print("\nH3: Spearman vs DSRS total_score (H3 predicts NEGATIVE rho for patient excess;")
    print("caregiver is the control and should be ~0 if the effect is about the patient)")
    print("raw = similarity to the prompt itself; excess = raw minus similarity to the visit's other")
    print("prompts. If excess moves but raw doesn't, the change is in the baseline (how varied the")
    print("provider's prompts are), not in how closely the reply follows its prompt.")
    print(f"  {'feature':48s} {'all visits':34s} earliest visit per patient")
    cols = [c for c in df.columns if c.startswith("reply_")]
    for col in cols:
        print(f"  {col:48s} {rho(d[col], d['total_score']):34s} {rho(first[col], first['total_score'])}")

    print(f"\ncheck 1: controlling for talk-time ({' + '.join(TALK_CONTROLS)}; share of speaker blocks).")
    print("If rho shrinks toward 0, the feature is mostly re-measuring who talks more.")
    print(f"  {'feature':48s} {'all visits':34s} earliest visit per patient")
    for col in KEY_FEATURES:
        print(f"  {col:48s} {partial_rho(d[col], d['total_score'], d[TALK_CONTROLS]):34s} "
              f"{partial_rho(first[col], first['total_score'], first[TALK_CONTROLS])}")

    print("\ncheck 2: within each site. A pooled effect that vanishes within both sites can come")
    print("from the sites differing in both DSRS and the feature (e.g. provider style), not severity.")
    for site, g in d.groupby("site"):
        print(f"  {site}: {len(g)} visits, {g['record_id_clean'].nunique()} patients, "
              f"median DSRS {g['total_score'].median():.0f}")
    for col in KEY_FEATURES:
        site_means = "  ".join(f"{s} mean={g[col].mean():+.3f}" for s, g in d.groupby("site"))
        print(f"  {col}  ({site_means})")
        for site, g in d.groupby("site"):
            print(f"      {site:10s} {rho(g[col], g['total_score'])}")
    print("\nThese are exploratory tests: about 30 correlations in the main table alone, so treat")
    print("p values near .05 as leads; the cross-validated ablation in predict_dsrs.py is the real test.")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--model", default=EMBED_MODEL, help="sentence-transformers model (default: %(default)s)")
    ap.add_argument("--embeddings-cache", help="default: outputs/turn_embeddings for the default model, "
                    "outputs/turn_embeddings_<model> otherwise")
    ap.add_argument("--daseg-dir", default=DEFAULT_DASEG_DIR)
    ap.add_argument("--out", help="default: <data_dir>/derived/reply_relevance_<model>.csv")
    ap.add_argument("--stats", action="store_true", help="also print the H3 correlations with DSRS")
    args = ap.parse_args()

    cache_dir = resolve_cache_dir(args.model, args.embeddings_cache)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    idx = load_dataset(args.data_dir, verbose=False)
    idx = idx.assign(visit_id=idx["path"].map(visit_key))
    roles = load_roles(args.data_dir)
    das = daseg_index(args.daseg_dir)
    print(f"model={args.model}  cache={cache_dir}  DASeg files found: {len(das)}")

    rows, misaligned = [], 0
    for i, row in idx.iterrows():
        t = load_transcript(row["path"], roles)
        emb, _ = get_turn_embeddings(row["path"], roles, cache_dir, model)
        if len(emb) != len(t):
            misaligned += 1
            continue
        questions = question_turns(das[row["visit_id"]]) if row["visit_id"] in das else None
        blocks = merge_blocks(t, emb)
        feats = visit_features(reply_records(blocks, questions), questions is not None)
        feats.update({"visit_id": row["visit_id"], "daseg_available": int(questions is not None),
                      "embed_model": args.model})
        for r in RESPONDER_ROLES:
            feats[f"ctl_share_{r}"] = sum(b["role"] == r for b in blocks) / max(len(blocks), 1)
        rows.append(feats)
        print(f"\r  processed {i + 1}/{len(idx)} visits", end="", flush=True)
    print()
    if misaligned:
        print(f"  WARNING: skipped {misaligned} visits whose cached embeddings don't match the transcript "
              f"row count; delete those .npz files and rerun")

    df = pd.DataFrame(rows)
    out = args.out or os.path.join(args.data_dir, "derived", f"reply_relevance_{model_slug(args.model)}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {len(df)} visits x {df.shape[1]} columns -> {out}")

    if args.stats:
        run_stats(idx, df)

if __name__ == "__main__":
    main()