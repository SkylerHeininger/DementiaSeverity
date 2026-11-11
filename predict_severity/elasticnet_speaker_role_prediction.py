"""
Skyler Heininger
COSC 272
11/10/2026

This script uses ElasticNET to predict the DSRS of a session solely using the percent each speaker made in a session.

This is a simpler, yet very powerful approach. We both direct-fit and use 5-fold CV for this.

Within this script, Claude Sonnet 5.5 was used to help with graphing functions, some errors run into with tensor shapes, 
and helping to troubleshoot some issues ran into with the data loading (this changed a couple of times through the project, 
due to varying amounts of data made available, so Sonnet helped to iterate on existing data loading pipelines).
(this is the same across cnn_predict_sequence_da.py, elasticnet_predict_session_level_da.py, and elasticnet_speaker_role_prediction.py,
as there is a bit of repeated code among them)
"""

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Just looking at percentage that each person speaks
FEATURES = ["provider_pct", "patient_pct", "caregiver_pct"]
VISIT_STYLES = {
    "other": ("#4C72B0", "o"),
    "dyadic": ("#DD8452", "s"),
    "bilingual": ("#55A868", "^"),
    "dyadic and bilingual": ("#C44E52", "D"),
}
BAND_LABELS = ("0-18", "19-36", "37-54")
BAND_EDGES = (18.49, 36.49)


def normalize_id(value):
    # Grabs the Id from the filename
    tokens = [t for t in re.split(r"[^A-Z0-9]+", str(value).strip().upper()) if t]
    if not tokens:
        return ""

    if len(tokens) >= 2 and tokens[1].isdigit():
        num = int(tokens[1])
        return f"{tokens[0]}_{num:04d}" if tokens[0][0] == "D" else f"{tokens[0]}-{num:03d}"

    return "-".join(tokens)


def group_folds(groups, n_splits=5, seed=0):
    # Group by patient Id
    ids = np.unique(groups)
    np.random.default_rng(seed).shuffle(ids)
    fold_of = {g: i % n_splits for i, g in enumerate(ids)}
    fold = np.array([fold_of[g] for g in groups])

    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_splits)]


def fit_and_cross_validate(df, cols, cv, train_ok=None):
    # Does both direct fit and 5-fold cross-validation
    X = df[cols].to_numpy(dtype=float)
    y = df["total_score"].to_numpy(dtype=float)

    fit_rows = np.ones(len(y), dtype=bool) if train_ok is None else train_ok

    # Direct fit
    model = LinearRegression().fit(X[fit_rows], y[fit_rows])
    r2 = model.score(X[fit_rows], y[fit_rows])

    # CV
    cv_pred = np.zeros(len(y))
    fold_r2 = []
    for train, test in cv:
        fit_idx = train if train_ok is None else train[train_ok[train]]
        cv_pred[test] = LinearRegression().fit(X[fit_idx], y[fit_idx]).predict(X[test])
        test_eval = test[fit_rows[test]]
        fold_r2.append(r2_score(y[test_eval], cv_pred[test_eval]))
    fold_r2 = np.array(fold_r2)

    print(f"\ntotal_score: {' + '.join(cols)}")
    print(f"in-sample R^2 = {r2:.3f}")
    print(f"5-fold CV R^2: mean = {fold_r2.mean():.3f}, folds = {[round(float(s), 3) for s in fold_r2]}")
    print(f"pooled CV R^2 (all held-out predictions together) = {r2_score(y[fit_rows], cv_pred[fit_rows]):.3f}")
    return model, r2, fold_r2.mean(), cv_pred

def scatter_by_visit_type(ax, df, x_col, y_col):
    # For printing
    for kind, sub in df.groupby("visit_type"):
        color, marker = VISIT_STYLES[kind]
        ax.scatter(sub[x_col], sub[y_col], alpha=0.7, color=color, marker=marker)


def legend_handles(df):
    present = set(df["visit_type"])

    return [Line2D([0], [0], marker=marker, linestyle="", color=color, label=kind)
            for kind, (color, marker) in VISIT_STYLES.items() if kind in present]


def add_legend(ax, df, fontsize, loc, title=None):
    # For the run without dyadic/bilingual training, don't make a legend
    handles = [] if set(df["visit_type"]) == {"other"} else legend_handles(df)
    if not handles and title is None:
        return

    ax.legend(handles=handles, title=title, fontsize=fontsize, title_fontsize=fontsize, loc=loc, alignment="left")


def to_band(values):
    return np.digitize(values, BAND_EDGES, right=True)


def confusion_matrix(true, pred):
    matrix = np.zeros((len(BAND_LABELS), len(BAND_LABELS)), dtype=int)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def recalls(matrix):
    totals = matrix.sum(axis=1)
    return np.where(totals > 0, np.diag(matrix) / np.maximum(totals, 1), np.nan)


def majority_band_predictions(y, groups, train_ok=None):
    bands = to_band(y)
    pred = np.zeros(len(y), dtype=int)

    for train, test in group_folds(groups, 5, 0):
        if train_ok is not None:
            train = train[train_ok[train]]
        pred[test] = np.bincount(bands[train], minlength=len(BAND_LABELS)).argmax()

    return pred


def plot_confusions(entries, path):
    ncols = min(4, len(entries))
    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 4.1 * nrows), squeeze=False)

    for ax in axes.ravel()[len(entries):]:
        ax.axis("off")

    for ax, (name, matrix) in zip(axes.ravel(), entries):
        totals = matrix.sum(axis=1, keepdims=True)
        shares = np.divide(matrix, totals, out=np.zeros(matrix.shape), where=totals > 0)
        ax.imshow(shares, vmin=0, vmax=1, cmap="Blues")

        for i in range(len(BAND_LABELS)):
            for j in range(len(BAND_LABELS)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if shares[i, j] > 0.5 else "black")

        ax.set_xticks(range(len(BAND_LABELS)))
        ax.set_xticklabels(BAND_LABELS)
        ax.set_yticks(range(len(BAND_LABELS)))
        ax.set_yticklabels(BAND_LABELS)
        ax.set_xlabel("Predicted DSRS")
        ax.set_ylabel("True DSRS")
        ax.set_title(f"{textwrap.fill(name, 30)}\naccuracy {np.trace(matrix) / matrix.sum():.2f}, balanced {np.nanmean(recalls(matrix)):.2f}", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusions(y, groups, models, train_ok, hard, output_dir, prefix):
    headline = np.ones(len(y), dtype=bool) if train_ok is None else train_ok
    headline_label = "all visits in this run" if train_ok is None else "visits that are not dyadic or bilingual"
    bands = {"majority band baseline": majority_band_predictions(y, groups, train_ok)}
    bands.update({name: to_band(pred) for name, pred in models.items()})
    sets = [("", headline_label, headline)]

    if train_ok is not None and hard.any():
        sets.append(("_hard", "held-out dyadic and bilingual visits", hard))

    for suffix, label, mask in sets:
        true = to_band(y[mask])
        entries = [(name, confusion_matrix(true, pred[mask])) for name, pred in bands.items()]
        path = output_dir / f"{prefix}_confusion{suffix}.png"
        print(f"\nConfusion matrices, {label}: {int(mask.sum())} visits, true counts by band {BAND_LABELS}: {np.bincount(true, minlength=len(BAND_LABELS)).tolist()}")

        for name, matrix in entries:
            print(f"{name} | accuracy {np.trace(matrix) / matrix.sum():.3f} | balanced accuracy {np.nanmean(recalls(matrix)):.3f}")
            print(pd.DataFrame(matrix, index=[f"true {b}" for b in BAND_LABELS], columns=[f"pred {b}" for b in BAND_LABELS]).to_string())

        plot_confusions(entries, path)
        print(f"Wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict DSRS total score from speaker role shares.")
    ap.add_argument("--dataset", default="dataset.csv", help="Output of data_processing.data_loader.")
    ap.add_argument("--blacklist", default=None, help="Text file with one record id per line to exclude (e.g. D01_0002), at every timepoint.")
    ap.add_argument("--include-dyads", action="store_true", help="Keep sessions with no caregiver speech (patient-provider dyads), dropped by default.")
    ap.add_argument("--timepoints", nargs="+", default=None, help="Timepoints to use (e.g. T0 T2 T3, or 'all'). Default: all.")
    ap.add_argument("--exclude-bilingual", action="store_true", help="Drop records flagged is_bilingual in the dataset.")
    ap.add_argument("--train-without-hard", action="store_true", help="Train only on sessions that are not dyadic or bilingual, but keep those sessions and score on them (overrides --exclude-bilingual and the dyad drop).")
    ap.add_argument("--auc-cutoff", type=float, default=19, help="DSRS cutoff for the AUC reported for the all-features model.")
    ap.add_argument("--save-model", default=None, help="Path to save the all-features model for the app (e.g. models/speaker_roles.joblib).")
    ap.add_argument("--output-dir", default=".", help="Folder the plots are written into (created if missing).")
    args = ap.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "speech_features_vs_severity.png"
    cv_plot_path = output_dir / "speech_features_vs_severity_cv.png"
    prediction_path = output_dir / "speech_cv_predictions.csv"

    # Load data
    df = pd.read_csv(args.dataset)
    required = [*FEATURES, "total_score", "site", "timepoint", "record_id"]
    if args.exclude_bilingual or args.train_without_hard:
        required.append("is_bilingual")
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.dataset} is missing column(s) {missing}.")

    # Remove blacklisted IDs (bad diarization or sus DSRS)
    if args.blacklist:
        blacklist_path = Path(args.blacklist)
        if not blacklist_path.is_file():
            raise SystemExit(f"blacklist file not found: {blacklist_path}")
        
        blacklist = {normalize_id(line) for line in blacklist_path.read_text().splitlines()} - {""}
        ids = df["record_id"].map(normalize_id)
        on_blacklist = ids.isin(blacklist)
        
        print(f"\nBlacklist: {len(blacklist)} id(s) from {blacklist_path}. Their rows are removed at every timepoint.")
        for rid, n in ids[on_blacklist].value_counts().sort_index().items():
            print(f"excluded: {rid} ({n} row(s))")
        for rid in sorted(blacklist - set(ids)):
            print(f"blacklisted id not found in the dataset: {rid}")

        df = df[~on_blacklist].reset_index(drop=True)
    
    if args.exclude_bilingual and not args.train_without_hard:
        # I believe this is effectively code bloat at this point
        n_before = len(df)
        df = df[~df["is_bilingual"].astype(bool)].reset_index(drop=True)
        print(f"Excluded {n_before - len(df)} bilingual record(s)")
    
    if not args.include_dyads and not args.train_without_hard:
        dyads = df["caregiver_pct"] == 0
        for _, row in df[dyads].iterrows():
            print(f"excluded, no caregiver turns (patient-provider dyad): {row['record_id']} {row['timepoint']} "
                  f"{row.get('visit', '')} {row.get('transcript_files', '')}")
        print(f"Excluded {int(dyads.sum())} session(s) with no caregiver turns")
        df = df[~dyads].reset_index(drop=True)

    # Grab timepoints
    if args.timepoints and "all" not in [t.lower() for t in args.timepoints]:
        wanted = [t.upper() for t in args.timepoints]
        unknown = sorted(set(wanted) - set(df["timepoint"].unique()))
        if unknown:
            raise SystemExit(f"--timepoints {unknown} not in the dataset (found: {sorted(df['timepoint'].unique())}).")
        
        df = df[df["timepoint"].isin(wanted)].reset_index(drop=True)
    
    if df["record_id"].nunique() < 5:
        raise SystemExit(f"Only {df['record_id'].nunique()} patient(s) -- need at least 5 for 5-fold cross-validation.")

    print(f"Loaded {len(df)} row(s) from {df['record_id'].nunique()} patient(s) in {args.dataset}")
    print(pd.crosstab(df["site"], df["timepoint"]).to_string())

    # some double checks (I believe some of this has become bloat over time too)
    dyad = (df["caregiver_pct"] == 0).to_numpy()
    bilingual = df["is_bilingual"].astype(bool).to_numpy() if "is_bilingual" in df.columns else np.zeros(len(df), dtype=bool)
    hard = dyad | bilingual
    df["visit_type"] = np.select([dyad & bilingual, dyad, bilingual], ["dyadic and bilingual", "dyadic", "bilingual"], default="other")
    train_ok = ~hard if args.train_without_hard else None

    print(f"Dyadic sessions: {int(dyad.sum())}, bilingual sessions: {int(bilingual.sum())}"+ (", used for scoring only, not for training"if args.train_without_hard else ""))
    df["site_is_nyu"] = (df["site"].astype(str).str.lower() == "nyu").astype(float)
    eval_mask = np.ones(len(df), dtype=bool) if train_ok is None else train_ok
    plot_sets = [("", eval_mask)]

    if train_ok is not None and hard.any():
        plot_sets.append(("_hard", hard))

    cv = group_folds(df["record_id"].to_numpy())

    truth = df["total_score"].to_numpy(dtype=float)
    panels = {col: fit_and_cross_validate(df, [col], cv, train_ok) for col in FEATURES}

    for suffix, mask in plot_sets:
        sub = df[mask]
        fig, axes = plt.subplots(1, len(FEATURES), figsize=(4.5 * len(FEATURES), 4.5))

        for ax, col in zip(axes, FEATURES):
            model, r2, cv_r2, panel_pred = panels[col]
            scatter_by_visit_type(ax, sub, col, "total_score")
            x_line = np.linspace(sub[col].min(), sub[col].max(), 100)
            ax.plot(x_line, model.predict(x_line.reshape(-1, 1)), color="firebrick", linewidth=1.5)
            note = f"n={int(mask.sum())}\nMAE={np.abs(panel_pred[mask] - truth[mask]).mean():.2f}" if suffix else f"R\u00b2={r2:.2f}\nCV R\u00b2={cv_r2:.2f}"
            ax.text(0.05, 0.95, note, transform=ax.transAxes,
                    ha="left", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
            ax.set_xlabel(col.replace("_", ""))
            ax.set_ylabel("Total DSRS score")
            add_legend(ax, sub, 8, "upper right")

        fig.tight_layout()
        path = plot_path.with_name(f"{plot_path.stem}{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\nWrote {path}")

    model, r2, cv_r2, cv_pred = fit_and_cross_validate(df, FEATURES, cv, train_ok)
    print(f"intercept = {model.intercept_:.3f}")
    for col, coef in zip(FEATURES, model.coef_):
        print(f"{col:<14} coef = {coef:.3f}")

    # Check diff between sites
    print("\nSite only")
    fit_and_cross_validate(df, ["site_is_nyu"], cv, train_ok)
    y_all = df["total_score"].to_numpy(dtype=float)
    y_eval = y_all[eval_mask]
    if 0 < (y_eval >= args.auc_cutoff).sum() < len(y_eval):
        print(f"\nAUC for DSRS >= {args.auc_cutoff:g} from the all-features CV predictions: {roc_auc_score(y_eval >= args.auc_cutoff, cv_pred[eval_mask]):.3f}")
    if hard.any():
        err = np.abs(cv_pred - y_all)
        print(f"MAE on dyadic or bilingual sessions ({int(hard.sum())}): {err[hard].mean():.2f}; on the rest ({int((~hard).sum())}): {err[~hard].mean():.2f}")

    # Save results and plotting
    save_confusions(y_all, df["record_id"].to_numpy(), {"speaker roles (provider, patient, caregiver shares)": cv_pred}, train_ok, hard, output_dir, "speech")

    df["cv_predicted"] = cv_pred
    predictions = df[[c for c in ("record_id", "visit", "site", "timepoint", "total_score", "cv_predicted") if c in df.columns]].copy()
    predictions["dyadic"] = dyad
    predictions["bilingual"] = bilingual
    predictions.to_csv(prediction_path, index=False)
    print(f"Wrote {prediction_path}")

    if args.save_model:
        X_all = df[FEATURES].to_numpy(dtype=float)
        y_all = df["total_score"].to_numpy(dtype=float)
        fit_rows = np.ones(len(y_all), dtype=bool) if train_ok is None else train_ok
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": make_pipeline(StandardScaler(), LinearRegression()).fit(X_all[fit_rows], y_all[fit_rows]),
            "features": FEATURES,
            "cv": {"r2": float(cv_r2), "mae": float(np.abs(df["cv_predicted"] - df["total_score"])[eval_mask].mean())},
            "n_rows": int(len(df)), "n_patients": int(df["record_id"].nunique()),
            "timepoints": sorted(df["timepoint"].unique().tolist()),
        }, save_path)
        print(f"Saved model to {save_path}")
    
    for suffix, mask in plot_sets:
        sub = df[mask]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        scatter_by_visit_type(ax, sub, "total_score", "cv_predicted")
        lo = min(sub["total_score"].min(), sub["cv_predicted"].min())
        hi = max(sub["total_score"].max(), sub["cv_predicted"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
        note = f"n={int(mask.sum())}\nMAE={np.abs(sub['cv_predicted'] - sub['total_score']).mean():.2f}" if suffix else f"CV R\u00b2={cv_r2:.2f}"
        ax.set_xlabel("True total DSRS score")
        ax.set_ylabel("5-fold CV predicted total DSRS score\n(provider + patient + caregiver)")
        add_legend(ax, sub, 9, "upper left", note)
        fig.tight_layout()
        path = cv_plot_path.with_name(f"{cv_plot_path.stem}{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
