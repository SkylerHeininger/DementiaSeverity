"""
Forrest Leung
11/2/26

Validation script for Phase 2 (permutation test and repeated grouped CV) of the Dementia Severity project.

To run:
python embeddings/phase2_validate.py "$DATA"
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from load_data import (TIMEPOINT_MONTHS, WINDOW_MONTHS, build_index, compute_folds,
                       find_transcripts, key as visit_key, load_dataset, load_severity, norm_id)
from predict_dsrs import (feature_columns, load_feature_table, metrics, run_baseline,
                          run_linear)

RNG_SEED = 0

# Shuffle scores between patients while keeping each patient's own visits together.
def permute_labels_by_patient(y, groups, rng):
    y_perm = y.copy()
    uniq = np.unique(groups)
    remap = dict(zip(uniq, rng.permutation(uniq)))
    for g, h in remap.items():
        src, dst = np.where(groups == h)[0], np.where(groups == g)[0]
        k = min(len(src), len(dst))
        y_perm[dst[:k]] = y[src[:k]]
    return y_perm

# Perform a single pooled out-of-fold prediction for each visit using a fixed Ridge alpha.
def pooled_predict(df, feats, alpha, y_col="total_score"):
    preds = pd.Series(np.nan, index=df.index)
    for f in sorted(df["fold"].dropna().unique()):
        train = df[df["fold"] != f]
        test_idx = df.index[df["fold"] == f]
        medians = train[feats].median()
        Xtr = train[feats].fillna(medians)
        Xte = df.loc[test_idx, feats].fillna(medians)
        scaler = StandardScaler().fit(Xtr)
        model = Ridge(alpha=alpha).fit(scaler.transform(Xtr), train[y_col].to_numpy(float))
        preds.loc[test_idx] = model.predict(scaler.transform(Xte))
    return preds

# Extract the Ridge alpha value from the best_params string.
def extract_alpha(best_params_str):
    m = re.search(r"'alpha':\s*(?:np\.float64\()?([0-9.eE+-]+)\)?", best_params_str)
    return float(m.group(1))

def permutation_test(df, feats, n_perm=200, seed=RNG_SEED):
    ridge_folds, _, _ = run_linear(df, feats, "ridge")
    alpha = float(np.median([extract_alpha(p) for p in ridge_folds["best_params"]]))

    y = df["total_score"].to_numpy(float)
    groups = df["record_id_clean"].to_numpy()
    obs_pred = pooled_predict(df, feats, alpha).to_numpy()
    obs = metrics(y, obs_pred)

    rng = np.random.default_rng(seed)
    null_rho, null_r2 = np.zeros(n_perm), np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = permute_labels_by_patient(y, groups, rng)
        d = df.copy()
        d["total_score"] = y_perm
        pred = pooled_predict(d, feats, alpha).to_numpy()
        m = metrics(y_perm, pred)
        null_rho[i], null_r2[i] = m["spearman_rho"], m["r2"]

    p_rho = (np.sum(np.abs(null_rho) >= abs(obs["spearman_rho"])) + 1) / (n_perm + 1)
    p_r2 = (np.sum(null_r2 >= obs["r2"]) + 1) / (n_perm + 1)
    return {
        "alpha_used": alpha, "n_perm": n_perm,
        "observed_rho": obs["spearman_rho"], "observed_r2": obs["r2"],
        "null_rho_mean": float(np.nanmean(null_rho)), "null_rho_std": float(np.nanstd(null_rho)),
        "null_r2_mean": float(np.nanmean(null_r2)), "null_r2_std": float(np.nanstd(null_r2)),
        "p_value_rho": float(p_rho), "p_value_r2": float(p_r2),
    }

# Perform repeated grouped cross-validation for Ridge and Elastic Net models, including ablation studies.
def repeated_cv(df, data_dir, probe_set, n_seeds=10, k=5, ablation_groups=("turn_dynamics", "topic_duration", "embedding_similarity", "all")):
    rows = []
    for seed in range(n_seeds):
        d = df.copy()
        # in memory: assign_folds would overwrite derived/fold_assignment.json once per seed
        d["fold"] = d["record_id_clean"].astype(str).map(compute_folds(d, k=k, seed=seed)).astype("Int64").to_numpy()
        feats = [c for c in feature_columns(d, "all", probe_set) if not d[c].isna().all()]

        b, _ = run_baseline(d)
        b["seed"], b["probe_set"] = seed, probe_set
        rows.append(b)
        r, _, _ = run_linear(d, feats, "ridge")
        r["seed"], r["probe_set"] = seed, probe_set
        rows.append(r)
        e, _, _ = run_linear(d, feats, "elastic_net")
        e["seed"], e["probe_set"] = seed, probe_set
        rows.append(e)

        for g in ablation_groups:
            gfeats = [c for c in feature_columns(d, g, probe_set) if not d[c].isna().all()]
            if gfeats:
                gm, _, _ = run_linear(d, gfeats, "ridge")
                gm["seed"], gm["probe_set"], gm["feature_group"] = seed, probe_set, g
                rows.append(gm)
        print(f"  repeated CV seed {seed + 1}/{n_seeds} done (probe_set={probe_set})", flush=True)
    return pd.concat(rows, ignore_index=True)

# Paired comparison: does adding a feature group to a base group improve held-out prediction?
# Both models see the SAME folds in every seed, so per-fold differences are paired.
PAIRED_COMPARISONS = [("talk_time", "talk_time+reply_relevance"),
                      ("talk_time", "talk_time+embedding_similarity"),
                      ("talk_time", "talk_time+probes_max"),     # probes alone, without alignment/coherence
                      ("talk_time", "talk_time+alignment"),      # the other half of embedding_similarity
                      ("talk_time", "talk_time+probes_rate"),    # length-controlled probes
                      ("talk_time", "talk_time+probes_role")]    # which speaker raises the symptom topics
PROBE_DEPENDENT = ("embedding_similarity", "probes_")

def corrected_ttest(diffs, test_frac):
    """Nadeau & Bengio (2003) corrected resampled t-test. Repeated-CV folds share training data,
    so a plain t-test on fold differences is far too confident; the (1/J + n_test/n_train)
    factor inflates the variance to account for that overlap."""
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    J = len(d)
    if J < 3 or d.std(ddof=1) == 0:
        return np.nan, np.nan
    se = np.sqrt((1 / J + test_frac / (1 - test_frac)) * d.var(ddof=1))
    t = d.mean() / se
    return float(t), float(2 * student_t.sf(abs(t), J - 1))

def paired_group_comparison(df, base, added, probe_set="orig", n_seeds=10, k=5):
    seed_rows, fold_rows = [], []
    for seed in range(n_seeds):
        d = df.copy()
        d["fold"] = d["record_id_clean"].astype(str).map(compute_folds(d, k=k, seed=seed)).astype("Int64")
        per_fold = {}
        for g in (base, added):
            feats = [c for c in feature_columns(d, g, probe_set) if not d[c].isna().all()]
            fm, _, preds = run_linear(d, feats, "ridge")
            per_fold[g] = fm.set_index("fold")
            seed_rows.append({"seed": seed, "group": g, "n_features": len(feats),
                              **metrics(preds["y_true"], preds["y_pred"])})   # pooled over the seed's folds
        for f in per_fold[base].index:
            fold_rows.append({"seed": seed, "fold": int(f),
                              **{f"delta_{m}": per_fold[added].at[f, m] - per_fold[base].at[f, m]
                                 for m in ("mae", "rmse", "r2")}})
        print(f"\r  seed {seed + 1}/{n_seeds}", end="", flush=True)   # short: a wrapped line breaks \r
    print("\r", end="")
    seeds, folds = pd.DataFrame(seed_rows), pd.DataFrame(fold_rows)
    seeds["comparison"] = folds["comparison"] = f"{added} vs {base}"
    seeds["probe_set"] = folds["probe_set"] = probe_set
    return seeds, folds

def summarize_paired(seeds, folds, base, added, k):
    s = seeds.pivot(index="seed", columns="group", values="r2")
    d_seed = s[added] - s[base]
    n_feats = seeds.groupby("group")["n_features"].first()
    print(f"  pooled R2 per seed: {base} ({n_feats[base]} feats) {s[base].mean():.3f} ± {s[base].std():.3f}  |  "
          f"{added} ({n_feats[added]} feats) {s[added].mean():.3f} ± {s[added].std():.3f}")
    print(f"  delta R2 (pooled): {d_seed.mean():+.3f}, {added} better in {(d_seed > 0).sum()}/{len(d_seed)} seeds")
    for m in ("r2", "rmse", "mae"):
        dm = folds[f"delta_{m}"]
        better = (dm > 0) if m == "r2" else (dm < 0)
        t, p = corrected_ttest(dm, 1 / k)
        print(f"  delta {m:4s} per fold: mean {dm.mean():+.3f}, better in {better.mean():.0%} of {len(dm)} folds, "
              f"corrected t={t:+.2f} p={p:.3g}")

# Perform confound checks on the final dataset, including site-level and coordinator-level analyses.
def confound_checks(idx_final, predictions_path=None):
    out = {}
    out["dsrs_by_site"] = idx_final.groupby("site")["total_score"].describe()[["count", "mean", "std"]]

    coord_rate = idx_final.groupby("site")["has_coordinator"].mean() if "has_coordinator" in idx_final else None
    if coord_rate is None:
        # If "has_coordinator" is not in idx_final, create an empty series for coord_rate.
        coord_rate = pd.Series(dtype=float)
    out["coordinator_rate_by_site"] = coord_rate

    if "has_coordinator" in idx_final:
        g = idx_final.groupby("has_coordinator")["total_score"]
        out["dsrs_by_coordinator_presence"] = g.describe()[["count", "mean", "std"]]
        a = idx_final.loc[idx_final["has_coordinator"] == 1, "total_score"]
        b = idx_final.loc[idx_final["has_coordinator"] == 0, "total_score"]
        if len(a) > 1 and len(b) > 1:
            out["coordinator_dsrs_mannwhitney_p"] = float(mannwhitneyu(a, b).pvalue)

    out["pct_non_english_by_site"] = idx_final.groupby("site")["pct_non_english"].describe()[["count", "mean", "max"]]

    if predictions_path and os.path.exists(predictions_path):
        pred = pd.read_csv(predictions_path)
        pred["error"] = pred["y_pred"] - pred["y_true"]
        meta = idx_final[["key", "site", "pct_non_english"]].rename(columns={"key": "visit_id"})
        pred = pred.merge(meta, on="visit_id", how="left")
        under_predicted = pred[(pred["model"] == "ridge") & (pred["pct_non_english"] >= 0.01)
                               & (pred["pct_non_english"] < 0.05)]
        out["under_predicted_1to5pct_by_site"] = under_predicted["site"].value_counts()

    return out

def ridge_with_site_covariate(df, probe_set):
    d = df.copy()
    d["site_is_nyu"] = (d["site"] == "NYU").astype(float)
    feats = [c for c in feature_columns(d, "all", probe_set) if not d[c].isna().all()] + ["site_is_nyu"]
    pooled, _, _ = run_linear(d, feats, "ridge")

    within_site = []
    for site in d["site"].dropna().unique():
        sub = d[d["site"] == site].copy()
        if sub["fold"].nunique() < 2 or sub["record_id_clean"].nunique() < 10:
            continue
        sfeats = [c for c in feature_columns(sub, "all", probe_set) if not sub[c].isna().all()]
        sm, _, _ = run_linear(sub, sfeats, "ridge")
        sm["site"] = site
        within_site.append(sm)
    return pooled, pd.concat(within_site, ignore_index=True) if within_site else pd.DataFrame()

# Perform matching recovery analysis to identify unmatched severity records and their causes.
def matching_recovery(data_dir):
    tr = find_transcripts(data_dir)
    sev = load_severity(data_dir)
    scored = sev.dropna(subset=["total_score"]).copy()
    scored["pid_norm"] = scored["record_id_clean"].map(norm_id)
    found = build_index(data_dir, verbose=False)

    matched_keys = set(found["key"])
    all_files = {}
    import glob as _glob
    for p in _glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True):
        if "derived" in p:
            continue
        from load_data import key as _key
        all_files.setdefault(_key(p), []).append(p)

    matched_rows = set(zip(found["record_id_clean"], found["Event Name"]))
    un_sev = scored[~scored.apply(lambda r: (r["record_id_clean"], r["Event Name"]) in matched_rows, axis=1)]

    tr_pids = set(tr["pid_norm"].dropna())
    causes_sev = {"no transcript for this patient at all": 0,
                 "patient has transcripts, but none in the date window": 0,
                 "matched_transcript names a file that isn't on disk": 0,
                 "other / unclear": 0}
    for _, r in un_sev.iterrows():
        if r["pid_norm"] not in tr_pids:
            causes_sev["no transcript for this patient at all"] += 1
        elif len(r["cand_keys"]) and not any(k in all_files for k in r["cand_keys"]):
            causes_sev["matched_transcript names a file that isn't on disk"] += 1
        else:
            target = TIMEPOINT_MONTHS.get(int(r["timepoint"])) if pd.notna(r["timepoint"]) else None
            months = tr.loc[tr["pid_norm"] == r["pid_norm"], "months"].dropna()
            if target is not None and len(months) and (months - target).abs().min() > WINDOW_MONTHS:
                causes_sev["patient has transcripts, but none in the date window"] += 1
            else:
                causes_sev["other / unclear"] += 1

    un_tr = tr[~tr["key"].isin(matched_keys)]
    sev_pids = set(scored["pid_norm"].dropna())
    causes_tr = {"no severity row for this patient at all": 0,
                "patient has severity rows, but none in this transcript's date window": 0,
                "other / unclear": 0}
    for _, r in un_tr.iterrows():
        if r["pid_norm"] not in sev_pids:
            causes_tr["no severity row for this patient at all"] += 1
        else:
            patient_rows = scored[scored["pid_norm"] == r["pid_norm"]]
            gaps = patient_rows["timepoint"].map(lambda tp: TIMEPOINT_MONTHS.get(int(tp)) if pd.notna(tp) else None)
            gaps = gaps.dropna()
            if len(gaps) and pd.notna(r["months"]) and (gaps - r["months"]).abs().min() > WINDOW_MONTHS:
                causes_tr["patient has severity rows, but none in this transcript's date window"] += 1
            else:
                causes_tr["other / unclear"] += 1

    return pd.Series(causes_sev, name="unmatched_severity_rows"), pd.Series(causes_tr, name="unmatched_transcripts")


# Perform a sanity check on segmenter boundaries by computing the F1 score between LLM and embedding boundaries.
def boundary_f1(llm_boundaries, embed_boundaries, tolerance=2):
    llm_b, emb_b = sorted(llm_boundaries), sorted(embed_boundaries)
    matched_llm = sum(1 for x in llm_b if any(abs(x - y) <= tolerance for y in emb_b))
    matched_emb = sum(1 for y in emb_b if any(abs(x - y) <= tolerance for x in llm_b))
    precision = matched_emb / max(len(emb_b), 1)   # of embedding boundaries, how many are near an LLM one
    recall = matched_llm / max(len(llm_b), 1)       # of LLM boundaries, how many are near an embedding one
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"n_llm": len(llm_b), "n_embedding": len(emb_b), "precision": precision,
           "recall": recall, "f1": f1}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--reply-relevance", help="reply_relevance.py output (H3 features), left-joined on visit_id")
    ap.add_argument("--probe-set", nargs="+", choices=["orig", "neutral", "both"], default=["orig"],
                    help="probe wording(s) for the embedding_similarity comparison in section 6; "
                         "several run one after another, e.g. --probe-set orig neutral (default: orig)")
    ap.add_argument("--comparisons", nargs="+",
                    help="only run section-6 comparisons whose added group contains one of these names, "
                         "e.g. --comparisons probes_ alignment (default: all)")
    ap.add_argument("--paired-only", action="store_true",
                    help="run only the paired feature-group comparison (section 6), skip sections 1-4")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    turn_dynamics_path = os.path.join(args.data_dir, "derived", "turn_dynamics_features.csv")
    embed_sim_path = os.path.join(args.data_dir, "derived", "embedding_similarity_features.csv")
    df = load_feature_table(turn_dynamics_path, None, embed_sim_path, args.reply_relevance)
    df = df.dropna(subset=["total_score", "fold"]).reset_index(drop=True)

    if args.paired_only:
        run_paired(df, args)
        return

    print("=== 1. permutation test (Ridge, orig probe set, current [embedding-boundary] features) ===")
    feats = [c for c in feature_columns(df, "all", "orig") if not df[c].isna().all()]
    perm = permutation_test(df, feats, n_perm=args.n_perm)
    for k, v in perm.items():
        print(f"  {k}: {v}")
    pd.Series(perm).to_csv(os.path.join(args.results_dir, "permutation_test.csv"))

    print("\n=== 2. repeated grouped CV (10 seeds x 5 folds), Ridge/EN/ablations, all 3 probe sets ===")
    rcv_parts = []
    for ps in ("orig", "neutral", "both"):
        rcv_parts.append(repeated_cv(df, args.data_dir, ps, n_seeds=args.n_seeds))
    rcv = pd.concat(rcv_parts, ignore_index=True)
    rcv.to_csv(os.path.join(args.results_dir, "repeated_cv.csv"), index=False)
    print(rcv.groupby(["probe_set", "model"])[["mae", "rmse", "r2", "spearman_rho"]].agg(["mean", "std"]).round(3))

    print("\n=== 3. confound checks ===")
    idx_final = load_dataset(args.data_dir, verbose=False)
    td = pd.read_csv(turn_dynamics_path)[["visit_id", "has_coordinator"]]
    idx_final = idx_final.assign(visit_id=idx_final["path"].map(visit_key))
    idx_final = idx_final.merge(td, on="visit_id", how="left")
    cc = confound_checks(idx_final, os.path.join(args.results_dir, "predictions.csv"))
    for k, v in cc.items():
        print(f"\n-- {k} --\n{v}")
        if isinstance(v, (pd.Series, pd.DataFrame)):
            v.to_csv(os.path.join(args.results_dir, f"confound_{k}.csv"))

    print("\nRidge with site covariate:")
    pooled, within_site = ridge_with_site_covariate(df, "orig")
    print(pooled[["fold", "mae", "r2", "spearman_rho"]].to_string(index=False))
    print("within-site:")
    if len(within_site):
        print(within_site[["site", "fold", "mae", "r2", "spearman_rho"]].to_string(index=False))
    pooled.to_csv(os.path.join(args.results_dir, "ridge_with_site_covariate.csv"), index=False)
    within_site.to_csv(os.path.join(args.results_dir, "ridge_within_site.csv"), index=False)

    print("\n=== 4. matching recovery ===")
    un_sev, un_tr = matching_recovery(args.data_dir)
    print("unmatched severity rows by cause:\n", un_sev.to_string())
    print("unmatched transcripts by cause:\n", un_tr.to_string())
    un_sev.to_csv(os.path.join(args.results_dir, "matching_recovery_severity.csv"))
    un_tr.to_csv(os.path.join(args.results_dir, "matching_recovery_transcripts.csv"))

    print("\n=== 5. segmenter sanity check: see compare_segmenters.py (needs LLM boundaries) ===")

    run_paired(df, args)


def run_paired(df, args, k=5):
    print(f"\n=== 6. paired feature-group comparison (Ridge, {args.n_seeds} seeds x {k} folds, same folds "
          f"for both models) ===")
    if args.reply_relevance:
        # compare on the same visits throughout: those that have reply features, so none are imputed
        n0 = len(df)
        df = df[df["daseg_available"].notna()].reset_index(drop=True)
        print(f"  restricted to visits with reply features: {len(df)}/{n0} visits, "
              f"{df['record_id_clean'].nunique()} patients")
    seed_parts, fold_parts = [], []
    for base, added in PAIRED_COMPARISONS:
        # probe wording only matters when the probes are in the comparison
        uses_probes = any(p in added for p in PROBE_DEPENDENT)
        wanted = getattr(args, "comparisons", None)
        if wanted and not any(c in added for c in wanted):
            continue
        probe_sets = args.probe_set if uses_probes else args.probe_set[:1]
        for ps in probe_sets:
            label = f"{added} vs {base}" + (f"  [probe_set={ps}]" if uses_probes else "")
            if not [c for c in feature_columns(df, added, ps) if c not in feature_columns(df, base, ps)]:
                print(f"  skipping {label}: none of the added group's features are in the table")
                continue
            print(f"\n  {label}")
            seeds, folds = paired_group_comparison(df, base, added, probe_set=ps, n_seeds=args.n_seeds, k=k)
            summarize_paired(seeds, folds, base, added, k)
            seed_parts.append(seeds)
            fold_parts.append(folds)
    if seed_parts:
        pd.concat(seed_parts).to_csv(os.path.join(args.results_dir, "paired_comparison_seeds.csv"), index=False)
        pd.concat(fold_parts).to_csv(os.path.join(args.results_dir, "paired_comparison_folds.csv"), index=False)
        print("  corrected t-test = Nadeau & Bengio (2003); with this few patients treat p as approximate")

if __name__ == "__main__":
    main()