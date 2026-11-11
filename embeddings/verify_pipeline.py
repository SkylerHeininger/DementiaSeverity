"""
Forrest Leung
11/2/26

This module implements verification checks for the dementia care visit pipeline, including data
and fold sanity, feature quality, and DSRS distribution analysis.

To run:
python embeddings/verify_pipeline.py "$DATA"
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import (annotate_visits, build_index, find_transcripts, load_dataset,
                       load_roles, load_severity, load_transcript, visit_language_stats)


# ---------- section 1: data / fold sanity ----------
def funnel_and_composition(data_dir):
    tr = find_transcripts(data_dir)
    sev = load_severity(data_dir)
    scored = sev.dropna(subset=["total_score"])
    idx0 = build_index(data_dir, verbose=False)   # matched to a transcript, pre-filter
    roles = load_roles(data_dir)

    has_cg, pct_non_en, has_pat, has_prov = [], [], [], []
    for _, row in idx0.iterrows():
        t = load_transcript(row["path"], roles)
        present = set(t["role"].unique())
        has_cg.append("caregiver" in present)
        has_pat.append("patient" in present)
        has_prov.append("provider" in present)
        share, _ = visit_language_stats(t)
        pct_non_en.append(share)
    idx0 = idx0.assign(has_caregiver=has_cg, has_patient=has_pat, has_provider=has_prov,
                       pct_non_english=pct_non_en)

    after_caregiver = idx0[idx0["has_caregiver"]]
    after_language = after_caregiver[after_caregiver["pct_non_english"] < 0.10]

    funnel = pd.DataFrame([
        {"step": "transcript CSVs on disk", "count": len(tr), "dropped": None, "reason": "raw files, pre-matching"},
        {"step": "severity survey rows", "count": len(sev), "dropped": None, "reason": "DSRS forms, all timepoints"},
        {"step": "...with a usable total_score", "count": len(scored), "dropped": len(sev) - len(scored),
         "reason": "incomplete DSRS form"},
        {"step": "matched to a transcript (visit-level)", "count": len(idx0), "dropped": len(scored) - len(idx0),
         "reason": "no transcript within ±3mo of the timepoint, or none for that patient"},
        {"step": "after caregiver filter", "count": len(after_caregiver), "dropped": len(idx0) - len(after_caregiver),
         "reason": "no caregiver turns at all"},
        {"step": "after language filter", "count": len(after_language), "dropped": len(after_caregiver) - len(after_language),
         "reason": "≥10% of judged words in non-English turns, or unreadable"},
    ])

    idx0["composition"] = np.select(
        [idx0["has_patient"] & idx0["has_caregiver"] & idx0["has_provider"],
         ~idx0["has_patient"] & idx0["has_caregiver"] & idx0["has_provider"]],
        ["triadic", "dyadic_caregiver_provider"], default="other")
    return funnel, idx0


def timepoint_coverage(idx0, idx_final):
    cov = idx_final["timepoint"].value_counts().sort_index()
    team_rejected_note = ("the pipeline log reports N pairings where the team's own "
                          "matched_transcript assignment put a T2/T3 survey outside the "
                          "±3-month window and re-matched by nearest date instead")
    gaps = (idx_final.assign(target=idx_final["timepoint"].map({0: 0, 2: 6, 3: 12}))
                     .assign(gap=lambda d: (d["months"] - d["target"]).abs()))
    bad_gaps = gaps[gaps["gap"] > 3]
    return cov, gaps.groupby("timepoint")["gap"].describe()[["count", "mean", "50%", "max"]], len(bad_gaps)


def fold_integrity(idx_final):
    per_patient_folds = idx_final.groupby("record_id_clean")["fold"].nunique()
    ok = bool((per_patient_folds <= 1).all())
    visits_per_fold = idx_final["fold"].value_counts().sort_index()
    patients_per_fold = idx_final.groupby("fold")["record_id_clean"].nunique().sort_index()
    return ok, visits_per_fold, patients_per_fold


def dsrs_distribution(idx_final, out_dir):
    overall = idx_final["total_score"].describe()
    per_fold = idx_final.groupby("fold")["total_score"].describe()[["count", "mean", "std", "min", "max"]]
    overall_mean, overall_std = overall["mean"], overall["std"]
    skewed = per_fold[(per_fold["mean"] - overall_mean).abs() > overall_std]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(idx_final["total_score"], bins=15, color="#4C72B0", edgecolor="white")
    axes[0].set_title("DSRS total_score, all visits")
    axes[0].set_xlabel("total_score"); axes[0].set_ylabel("count")
    for f in sorted(idx_final["fold"].unique()):
        axes[1].hist(idx_final.loc[idx_final["fold"] == f, "total_score"], bins=10, alpha=0.5, label=f"fold {f}")
    axes[1].set_title("DSRS total_score by fold"); axes[1].set_xlabel("total_score"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "dsrs_distribution.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return overall, per_fold, skewed, path


# ---------- section 2: feature quality ----------
def feature_quality(df, feats):
    rows = []
    for c in feats:
        s = df[c]
        rows.append({"feature": c, "pct_nan": float(s.isna().mean()),
                    "nunique": int(s.nunique(dropna=True)),
                    "std": float(s.std()) if s.notna().any() else np.nan,
                    "cv": float(s.std() / abs(s.mean())) if s.notna().any() and s.mean() not in (0, np.nan) else np.nan})
    q = pd.DataFrame(rows)
    constant = q[(q["nunique"] <= 1)]
    near_constant = q[(q["nunique"] > 1) & (q["nunique"] <= 3)]
    mostly_nan = q[q["pct_nan"] > 0.20]

    corr = df[feats].corr().abs()
    pairs = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r > 0.95:
                pairs.append({"feature_a": a, "feature_b": b, "abs_corr": float(r)})
    return q, constant, near_constant, mostly_nan, pd.DataFrame(pairs).sort_values(
        "abs_corr", ascending=False) if pairs else pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])


# ---------- section 4: residual analysis ----------
def residual_analysis(predictions_path, idx_final, idx0, out_dir):
    pred = pd.read_csv(predictions_path)
    pred["error"] = pred["y_pred"] - pred["y_true"]   # + = over-predicted, - = under-predicted

    # flag_non_english is False for every visit that survived the language filter (that's what
    # the filter means), so it carries no signal here; bin the underlying continuous share instead.
    meta = idx_final[["key", "timepoint", "pct_non_english"]].rename(columns={"key": "visit_id"})
    meta["language_bin"] = pd.cut(meta["pct_non_english"], [-0.001, 0.01, 0.05, 0.10],
                                  labels=["<1%", "1-5%", "5-10%"])
    comp = idx0[["key", "composition"]].rename(columns={"key": "visit_id"})
    pred = pred.merge(meta, on="visit_id", how="left").merge(comp, on="visit_id", how="left")

    models = [m for m in pred["model"].unique() if m != "baseline_mean"]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        sub = pred[pred["model"] == m]
        ax.scatter(sub["y_true"], sub["y_pred"], alpha=0.6, s=18)
        lim = [0, max(sub["y_true"].max(), sub["y_pred"].max()) + 2]
        ax.plot(lim, lim, "--", color="gray", linewidth=1)
        ax.set_title(m); ax.set_xlabel("actual DSRS"); ax.set_ylabel("predicted DSRS")
        ax.set_xlim(lim); ax.set_ylim(lim)
    fig.tight_layout()
    path = os.path.join(out_dir, "predicted_vs_actual.png")
    fig.savefig(path, dpi=110); plt.close(fig)

    by_timepoint = pred.groupby(["model", "timepoint"])["error"].agg(["mean", lambda s: s.abs().mean(), "count"])
    by_timepoint.columns = ["mean_signed_error", "mae", "n"]
    by_composition = pred.groupby(["model", "composition"])["error"].agg(["mean", lambda s: s.abs().mean(), "count"])
    by_composition.columns = ["mean_signed_error", "mae", "n"]
    by_language = pred.groupby(["model", "language_bin"], observed=True)["error"].agg(
        ["mean", lambda s: s.abs().mean(), "count"])
    by_language.columns = ["mean_signed_error", "mae", "n"]

    pred["dsrs_tercile"] = pred.groupby("model")["y_true"].transform(
        lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"], duplicates="drop"))
    by_tercile = pred.groupby(["model", "dsrs_tercile"], observed=True)["error"].agg(["mean", "count"])
    by_tercile.columns = ["mean_signed_error", "n"]
    return path, by_timepoint, by_composition, by_language, by_tercile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--predictions", help="predict_dsrs.py's predictions.csv, for section 4")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("=== section 1: data / fold sanity ===")
    funnel, idx0 = funnel_and_composition(args.data_dir)
    print(funnel.to_string(index=False))

    idx_final = load_dataset(args.data_dir, verbose=False)
    cov, gaps, n_bad_gaps = timepoint_coverage(idx0, idx_final)
    print("\ntimepoint coverage (final visits):\n", cov.to_string())
    print("\nmonths-from-target by timepoint:\n", gaps.round(2).to_string())
    print(f"visits matched outside the ±3-month window after all checks: {n_bad_gaps}")

    ok, vpf, ppf = fold_integrity(idx_final)
    print(f"\nfold integrity (no patient split across folds): {'PASS' if ok else 'FAIL'}")
    print("visits per fold:\n", vpf.to_string())
    print("patients per fold:\n", ppf.to_string())

    overall, per_fold, skewed, hist_path = dsrs_distribution(idx_final, args.results_dir)
    print("\nDSRS total_score, overall:\n", overall.round(2).to_string())
    print("\nper fold:\n", per_fold.round(2).to_string())
    if len(skewed):
        print(f"\nFLAGGED folds (mean > 1 overall std from the overall mean): {list(skewed.index)}")
    print(f"saved {hist_path}")

    comp = idx0.merge(idx_final[["key", "fold"]], on="key", how="inner")
    comp_table = pd.crosstab(comp["fold"], comp["composition"])
    print("\nvisit composition (triadic vs dyadic) per fold:\n", comp_table.to_string())

    idx_final.drop(columns=["path"]).to_csv(os.path.join(args.results_dir, "idx_final_metadata.csv"), index=False)
    funnel.to_csv(os.path.join(args.results_dir, "funnel.csv"), index=False)
    comp_table.to_csv(os.path.join(args.results_dir, "composition_by_fold.csv"))

    print("\n=== section 2: feature quality ===")
    td = pd.read_csv(os.path.join(args.data_dir, "derived", "turn_dynamics_features.csv"))
    es_path = os.path.join(args.data_dir, "derived", "embedding_similarity_features.csv")
    df = td.merge(pd.read_csv(es_path), on="visit_id", how="left") if os.path.exists(es_path) else td
    exclude = {"visit_id", "record_id_clean", "fold", "site", "timepoint", "total_score",
              "severity_category", "boundary_source"}
    feats = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    q, constant, near_constant, mostly_nan, corr_pairs = feature_quality(df, feats)
    print(f"{len(feats)} numeric features checked")
    print("\nconstant features:\n", constant.to_string(index=False) if len(constant) else "  (none)")
    print("\nnear-constant features (<=3 distinct values):\n",
          near_constant.to_string(index=False) if len(near_constant) else "  (none)")
    print("\nmostly-NaN features (>20% missing):\n",
          mostly_nan.to_string(index=False) if len(mostly_nan) else "  (none)")
    print(f"\nfeature pairs with |corr| > 0.95: {len(corr_pairs)}")
    if len(corr_pairs):
        print(corr_pairs.to_string(index=False))
    q.to_csv(os.path.join(args.results_dir, "feature_quality.csv"), index=False)
    corr_pairs.to_csv(os.path.join(args.results_dir, "feature_correlations_high.csv"), index=False)

    if args.predictions:
        print("\n=== section 4: residual analysis ===")
        scatter_path, by_tp, by_comp, by_lang, by_tercile = residual_analysis(
            args.predictions, idx_final, idx0, args.results_dir)
        print(f"saved {scatter_path}")
        print("\nerror by timepoint (mean_signed_error: + over-predicts, - under-predicts):\n",
              by_tp.round(2).to_string())
        print("\nerror by visit composition:\n", by_comp.round(2).to_string())
        print("\nerror by non-English word share (within the visits that passed the filter):\n",
              by_lang.round(2).to_string())
        print("\nerror by DSRS tercile (regression-to-the-mean check):\n", by_tercile.round(2).to_string())
        for name, tbl in (("error_by_timepoint", by_tp), ("error_by_composition", by_comp),
                         ("error_by_language", by_lang), ("error_by_dsrs_tercile", by_tercile)):
            tbl.to_csv(os.path.join(args.results_dir, f"{name}.csv"))


if __name__ == "__main__":
    main()
