"""
Forrest Leung
10/26/26

Embedding-based features from transcript text, and this track's exploratory corpus-level stats
This file owns the probe-based features used by predict_dsrs.py.

To run: 
python embeddings/embedding_similarity.py "$DATA"
python embeddings/embedding_similarity.py "$DATA" --stats (To explore corpus stats)
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict

from load_data import EMBED_MODEL
from load_data import get_turn_embeddings, key as visit_key
from load_data import baseline_visits, load_dataset, load_roles, load_transcript

RNG = np.random.default_rng(0)
MIN_WORDS = 4

PROBES_ORIG = {
    "memory_score": "forgets recent events, repeats the same questions, cannot remember things",
    "speech_language_score": "trouble finding words, hard to follow a conversation",
    "recognition_score": "does not recognize family members or familiar people",
    "orientation_time_score": "confused about the date, the day of the week, or the season",
    "orientation_place_score": "gets lost, does not know where they are",
    "decisions_score": "cannot make decisions or handle money and appointments",
    "personal_care_score": "needs help bathing, dressing and with hygiene",
    "eating_score": "needs help eating, poor appetite, trouble swallowing",
    "urination_bowels_score": "incontinence, accidents, loss of bladder or bowel control",
    "mobility_score": "falls, unsteady walking, uses a walker or wheelchair",
}
PROBES_NEUTRAL = {
    "memory_score": "talking about recent events, asking the same question more than once",
    "speech_language_score": "finding the right words, following along with the conversation",
    "recognition_score": "recognizing family members and familiar people",
    "orientation_time_score": "knowing the date, the day of the week, or the season",
    "orientation_place_score": "knowing where they are, finding their way around",
    "decisions_score": "making decisions, handling money and appointments",
    "personal_care_score": "bathing, dressing, and personal hygiene",
    "eating_score": "eating meals, appetite, swallowing",
    "urination_bowels_score": "bladder and bowel control",
    "mobility_score": "walking, balance, use of a walker or wheelchair",
}
PROBE_SETS = {"orig": PROBES_ORIG, "neutral": PROBES_NEUTRAL}

### Shared loading
def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

# idx and per-visit substantive-turn embeddings/roles/text/word-count from the shared cache
def load_corpus(data_dir, cache_dir, model):
    idx = load_dataset(data_dir, verbose=False)
    roles = load_roles(data_dir)
    visits = {}
    for _, row in idx.iterrows():
        t = load_transcript(row["path"], roles)
        emb, _ = get_turn_embeddings(row["path"], roles, cache_dir, model)
        sub = (t["word_count"] >= MIN_WORDS).to_numpy()
        visits[visit_key(row["path"])] = {
            "emb": emb[sub], "role": t.loc[sub, "role"].to_numpy(),
            "text": " ".join(t.loc[sub, "text"]), "n_words": int(t["word_count"].sum()),
        }
    idx = idx.assign(visit_id=idx["path"].map(visit_key))
    return idx, visits

# Per-visit features (predict_dsrs.py's embedding_similarity group)
def probe_embeddings(model):
    return {name: model.encode(list(probes.values()), normalize_embeddings=True)
            for name, probes in PROBE_SETS.items()}

# Compute per-visit probe scores
def visit_probe_scores(emb, probe_embs):
    feats = {}
    for set_name, P in probe_embs.items():
        keys = list(PROBE_SETS[set_name].keys())
        sims = emb @ P.T if len(emb) else np.zeros((0, len(keys)))
        for j, k in enumerate(keys):
            feats[f"probe_{set_name}_{k}"] = float(sims[:, j].max()) if len(emb) else np.nan
    return feats

# Length-controlled and per-speaker probe features.
PROBE_HIT_PERCENTILE = 95
PROBE_ROLES = ("patient", "caregiver", "provider")

# Compute corpus-wide probe hit thresholds
def probe_thresholds(visits, probe_embs, pct=PROBE_HIT_PERCENTILE):
    out = {}
    for set_name, P in probe_embs.items():
        sims = np.vstack([v["emb"] @ P.T for v in visits.values() if len(v["emb"])])
        mu, sd = sims.mean(axis=0), sims.std(axis=0) + 1e-12
        out[set_name] = {"per_probe": np.percentile(sims, pct, axis=0), "mu": mu, "sd": sd,
                         "any": float(np.percentile(((sims - mu) / sd).max(axis=1), pct))}
    return out

def visit_probe_rates(emb, role, probe_embs, thresholds):
    feats = {}
    for set_name, P in probe_embs.items():
        keys = list(PROBE_SETS[set_name].keys())
        th = thresholds[set_name]
        sims = emb @ P.T if len(emb) else np.zeros((0, len(keys)))
        hits = sims >= th["per_probe"]
        for j, k in enumerate(keys):
            feats[f"prate_{set_name}_{k}"] = float(hits[:, j].mean()) if len(emb) else np.nan
        any_hit = (((sims - th["mu"]) / th["sd"]).max(axis=1) >= th["any"]) if len(emb) else np.zeros(0, bool)
        for r in PROBE_ROLES:
            m = role == r
            feats[f"prole_{set_name}_{r}_any"] = float(any_hit[m].mean()) if m.any() else np.nan
    return feats

def visit_features(v, probe_embs, thresholds=None):
    feats = visit_probe_scores(v["emb"], probe_embs)
    feats["n_substantive_turns"] = len(v["emb"])   # length covariate for --probe-checks; not a model feature
    if thresholds is not None:
        feats.update(visit_probe_rates(v["emb"], v["role"], probe_embs, thresholds))
    pat = v["emb"][v["role"] == "patient"]
    cg = v["emb"][v["role"] == "caregiver"]
    if len(pat) and len(cg):
        feats["patient_caregiver_alignment"] = float(unit(pat.mean(0)) @ unit(cg.mean(0)))
    else:
        feats["patient_caregiver_alignment"] = np.nan
    feats["patient_coherence"] = (float(np.mean(np.sum(pat[:-1] * pat[1:], axis=1)))
                                  if len(pat) >= 3 else np.nan)
    return feats

def build_feature_table(idx, visits, probe_embs):
    thresholds = probe_thresholds(visits, probe_embs)
    rows = []
    for i, row in idx.iterrows():
        feats = visit_features(visits[row["visit_id"]], probe_embs, thresholds)
        feats["visit_id"] = row["visit_id"]
        rows.append(feats)
        print(f"\r  processed {i + 1}/{len(idx)} visits", end="", flush=True)
    print()
    return pd.DataFrame(rows)

# Exploratory corpus-level stats (--stats)
def within_vs_between(S, labels):
    iu = np.triu_indices(len(labels), 1)
    same = (labels[:, None] == labels[None, :])[iu]
    w, b = S[iu][same], S[iu][~same]
    return w.mean(), b.mean(), mannwhitneyu(w, b, alternative="greater").pvalue

# Do visits close in embedding space have close severity scores?
def mantel(S, y, groups, n_perm=2000):
    iu = np.triu_indices(len(y), 1)
    d_emb = (1 - S)[iu]
    r0 = spearmanr(d_emb, np.abs(y[:, None] - y[None, :])[iu]).correlation
    uniq = np.unique(groups)
    null = []
    for _ in range(n_perm):
        remap = dict(zip(uniq, RNG.permutation(uniq)))
        y_perm = y.copy()
        for g, h in remap.items():
            src, dst = np.where(groups == h)[0], np.where(groups == g)[0]
            k = min(len(src), len(dst))
            y_perm[dst[:k]] = y[src[:k]]
        null.append(spearmanr(d_emb, np.abs(y_perm[:, None] - y_perm[None, :])[iu]).correlation)
    return r0, (np.sum(np.array(null) >= r0) + 1) / (n_perm + 1)

def paired_cosine(A, B):
    ok = ~np.isnan(A).any(1) & ~np.isnan(B).any(1)
    return np.where(ok, np.sum(np.nan_to_num(A) * np.nan_to_num(B), axis=1), np.nan)

def cv_predict(X, y, groups):
    n_splits = min(5, len(np.unique(groups)))
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 20)), X, y,
                             groups=groups, cv=GroupKFold(n_splits))
    return spearmanr(pred, y).correlation

def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    r = spearmanr(a[ok], b[ok])
    return f"rho={r.correlation:+.3f} p={r.pvalue:.3g} n={ok.sum()}"

def run_stats(idx, visits, df):
    y = idx["total_score"].to_numpy(float)
    labels = idx["severity_category"].astype(str).to_numpy()
    groups = idx["record_id_clean"].astype(str).to_numpy()
    visit_ids = idx["visit_id"].tolist()
    n_words = np.array([visits[v]["n_words"] for v in visit_ids], float)
    print(f"{len(idx)} visits, {len(np.unique(groups))} patients")
    print("length confound check (n_words vs total_score):", rho(n_words, y))

    # Lexical baseline: TF-IDF cosine over each visit's concatenated substantive text
    texts = [visits[v]["text"] for v in visit_ids]
    T = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english").fit_transform(texts)
    S_tfidf = (T @ T.T).toarray()
    print("\n== TF-IDF (lexical) ==")
    print("within/between category: %.3f / %.3f  p=%.3g" % within_vs_between(S_tfidf, labels))
    print("Mantel r=%.3f p=%.3g" % mantel(S_tfidf, y, groups))
    print("CV ridge rho:", round(cv_predict(T.toarray(), y, groups), 3))

    # Doc-level embedding: Mean of a visit's own substantive-turn embeddings, then mean-centered across the corpus.
    dim = next(v["emb"].shape[1] for v in visits.values() if len(v["emb"]))
    doc = np.vstack([unit(visits[v]["emb"].mean(0)) if len(visits[v]["emb"]) else np.zeros(dim)
                     for v in visit_ids])
    doc = unit(doc - doc.mean(0))
    S = doc @ doc.T
    print("\n== embeddings (doc-level, mean-centered) ==")
    print("within/between category: %.3f / %.3f  p=%.3g" % within_vs_between(S, labels))
    print("Mantel r=%.3f p=%.3g" % mantel(S, y, groups))
    print("CV ridge rho (whole visit):", round(cv_predict(doc, y, groups), 3))

    role_emb = {}
    for role in ("patient", "caregiver"):
        arr = np.full((len(idx), dim), np.nan)
        for i, v in enumerate(visit_ids):
            rm = visits[v]["emb"][visits[v]["role"] == role]
            if len(rm):
                arr[i] = unit(rm.mean(0))
        role_emb[role] = arr
        ok = ~np.isnan(arr).any(1)
        if ok.sum() > 10:
            centered = unit(arr[ok] - arr[ok].mean(0))
            print(f"CV ridge rho ({role} only):", round(cv_predict(centered, y[ok], groups[ok]), 3))
    print("patient-caregiver alignment vs severity:",
          rho(paired_cosine(role_emb["patient"], role_emb["caregiver"]), y))
    print("patient local coherence vs severity:", rho(df["patient_coherence"].to_numpy(float), y))

    for set_name in PROBE_SETS:
        print(f"\ndomain probe vs matching subscore ({set_name} wording):")
        for k in PROBE_SETS[set_name]:
            if k in idx:
                print(f"  {k:24s}", rho(df[f"probe_{set_name}_{k}"], idx[k].to_numpy(float)))

# Visit length and speaker checks on the probe features
def probe_checks(idx, df):
    from reply_relevance import partial_rho
    d = idx[["visit_id", "record_id_clean", "timepoint", "total_score"]].merge(df, on="visit_id")
    first = baseline_visits(d)
    n_sub, y = d["n_substantive_turns"], d["total_score"]
    print(f"\n{len(d)} visits, {d['record_id_clean'].nunique()} patients; hit threshold = "
          f"{PROBE_HIT_PERCENTILE}th percentile of each probe's similarity over all substantive turns")
    print("length vs DSRS:", rho(n_sub, y))

    for set_name in PROBE_SETS:
        print(f"\n== check A: visit length ({set_name} wording) ==")
        print("max should track length (more turns, higher max); the rate should not. If 'max | length'")
        print("stays near 'max ~ DSRS', length isn't what drives the probe signal.")
        print(f"  {'probe':24s} {'max ~ length':30s} {'rate ~ length':30s} {'max ~ DSRS':30s} "
              f"{'max ~ DSRS | length':30s} rate ~ DSRS")
        for k in PROBE_SETS[set_name]:
            mx, rt = d[f"probe_{set_name}_{k}"], d[f"prate_{set_name}_{k}"]
            print(f"  {k:24s} {rho(mx, n_sub):30s} {rho(rt, n_sub):30s} {rho(mx, y):30s} "
                  f"{partial_rho(mx, y, n_sub):30s} {rho(rt, y)}")

        print(f"\n== check B: who raises the symptom topics ({set_name} wording) ==")
        print("share of each speaker's substantive turns that hit any probe; it is normalized by that")
        print("speaker's own turns, so it isn't just talk-time")
        print(f"  {'speaker':10s} {'mean rate':10s} {'~ DSRS, all visits':30s} earliest visit per patient")
        for r in PROBE_ROLES:
            col = f"prole_{set_name}_{r}_any"
            print(f"  {r:10s} {d[col].mean():<10.3f} {rho(d[col], y):30s} {rho(first[col], first['total_score'])}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--embeddings-cache", default="outputs/turn_embeddings")
    ap.add_argument("--out", help="defaults to <data_dir>/derived/embedding_similarity_features.csv")
    ap.add_argument("--stats", action="store_true", help="also print exploratory corpus-level stats")
    ap.add_argument("--probe-checks", action="store_true",
                    help="print the visit-length and per-speaker checks on the probe features")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    probe_embs = probe_embeddings(model)

    idx, visits = load_corpus(args.data_dir, args.embeddings_cache, model)

    df = build_feature_table(idx, visits, probe_embs)
    out = args.out or os.path.join(args.data_dir, "derived", "embedding_similarity_features.csv")
    df.to_csv(out, index=False)
    print(f"saved {len(df)} visits x {df.shape[1]} columns -> {out}")

    if args.probe_checks:
        probe_checks(idx, df)
    if args.stats:
        print()
        run_stats(idx, visits, df)

if __name__ == "__main__":
    main()