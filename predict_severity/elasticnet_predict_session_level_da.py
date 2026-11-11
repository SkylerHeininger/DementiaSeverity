import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VISIT_STYLES = {
    "other": ("#4C72B0", "o"),
    "dyadic": ("#DD8452", "s"),
    "bilingual": ("#55A868", "^"),
    "dyadic and bilingual": ("#C44E52", "D"),
}
BAND_LABELS = ("0-18", "19-36", "37-54")
BAND_EDGES = (18.49, 36.49)
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

EXTENDED_DA_GROUPS = {
    "questions": [
        "Wh-Question", "Yes-No-Question", "Open-Question", "Or-Clause",
        "Declarative-Yes-No-Question", "Rhetorical-Questions",
        "Tag-Question", "Declarative-Wh-Question",
    ],
    "answers": [
        "Yes-answers", "No-answers",
        "Affirmative-non-yes-answers", "Other-answers",
        "Negative-non-no-answers", "Dispreferred-answers", "Reject",
        "Response-Acknowledgement", "Hold-before-answer-agreement",
        "Agree-Accept",
    ],
    "statements": ["Statement-non-opinion", "Statement-opinion"],
    "backchannel": [
        "Acknowledge-Backchannel", "Backchannel-in-question-form",
        "Non-verbal", "Signal-non-understanding",
    ],
    "hedge": ["Hedge"],
    "social_ritual": [
        "Conventional-opening", "Conventional-closing",
        "Thanking", "Apology", "Appreciation", "Downplayer",
        "3rd-party-talk",
    ],
    "reformulation": ["Summarize/reformulate", "Repeat-phrase"],
    "elaboration": ["Collaborative-Completion", "Quotation"],
    "action": ["Action-directive", "Offers-Options-Commits"],
    "unintelligible": ["Uninterpretable", "Other"],
}
_DA_TO_GROUP = {da: grp for grp, members in EXTENDED_DA_GROUPS.items() for da in members}
DA_GROUPS = sorted(EXTENDED_DA_GROUPS.keys()) + ["other"]
ROLES = ("provider", "patient", "caregiver")
PAIR_COLS = [f"{r}_{g}_pct" for r in ROLES for g in DA_GROUPS]
TOTAL_COLS = [f"{r}_total_pct" for r in ROLES] + [f"{g}_total_pct" for g in DA_GROUPS]
SHARE_COLS = [f"{r}_{g}_of_all_pct" for r in ROLES for g in DA_GROUPS]


def map_da_to_group(da):
    return _DA_TO_GROUP.get(str(da).strip(), "other")


def group_folds(groups, n_splits=5, seed=0):
    # Group by patient Id
    ids = np.unique(groups)
    np.random.default_rng(seed).shuffle(ids)
    fold_of = {g: i % n_splits for i, g in enumerate(ids)}
    fold = np.array([fold_of[g] for g in groups])

    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_splits)]


def load_speaker_da_pct(path):
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return None
    if not {"role", "Proc_DA", "DA_number"}.issubset(df.columns):
        return None
    df = df[df["DA_number"] != ""]
    if df.empty:
        return None
    segments = df.groupby("DA_number", sort=False).agg(role=("role", "first"), da=("Proc_DA", "last"))
    segments["role"] = segments["role"].str.strip().str.lower()
    segments = segments[segments["role"].isin(ROLES)]
    if segments.empty:
        return None
    segments["group"] = segments["da"].map(map_da_to_group)
    counts = segments.groupby(["role", "group"]).size()
    role_counts = segments.groupby("role").size()
    group_counts = segments.groupby("group").size()
    out = {f"{r}_{g}_pct": float(counts.get((r, g), 0) / role_counts[r] * 100) if r in role_counts.index else 0.0 for r in ROLES for g in DA_GROUPS}
    out.update({f"{r}_total_pct": float(role_counts.get(r, 0) / len(segments) * 100) for r in ROLES})
    out.update({f"{g}_total_pct": float(group_counts.get(g, 0) / len(segments) * 100) for g in DA_GROUPS})
    out.update({f"{r}_{g}_of_all_pct": float(counts.get((r, g), 0) / len(segments) * 100) for r in ROLES for g in DA_GROUPS})
    return out


def build_da_dataset(dataset, daseg_dir):
    rows = []
    for rec in dataset.itertuples(index=False):
        feature_sets = []
        for name in str(rec.transcript_files).split(";"):
            name = name.strip()
            if not name:
                continue
            feats = load_speaker_da_pct(daseg_dir / f"{Path(name).stem}_with_predictions.csv")
            if feats is None:
                print(f"no daseg output: {rec.record_id} {rec.timepoint} {name}")
                continue
            feature_sets.append(feats)
        if not feature_sets:
            continue
        rows.append({
            "record_id": rec.record_id,
            "site": rec.site,
            "timepoint": rec.timepoint,
            "visit": getattr(rec, "visit", ""),
            "visit_date": getattr(rec, "visit_date", ""),
            "total_score": rec.total_score,
            "transcript_files": rec.transcript_files,
            "is_bilingual": getattr(rec, "is_bilingual", False),
            **{c: float(np.mean([f[c] for f in feature_sets])) for c in PAIR_COLS + TOTAL_COLS + SHARE_COLS},
        })
    columns = ["record_id", "site", "timepoint", "visit", "visit_date", "total_score", "transcript_files", "is_bilingual", *PAIR_COLS, *TOTAL_COLS, *SHARE_COLS]
    return pd.DataFrame(rows, columns=columns)


def make_model(name, n_features, max_features=None):
    steps = [StandardScaler()]

    if max_features is not None and n_features > max_features:
        steps.append(SelectKBest(f_regression, k=max_features))

    if name == "linear":
        steps.append(LinearRegression())
    elif name == "ridge":
        steps.append(RidgeCV(alphas=np.logspace(-2, 3, 20)))
    else:
        steps.append(ElasticNetCV(l1_ratio=L1_RATIOS, alphas=50, max_iter=100000, eps=1e-2))

    return make_pipeline(*steps)


def selected_columns(model, cols):
    selectors = [step for _, step in model.steps if isinstance(step, SelectKBest)]

    if not selectors:
        return cols

    return list(np.array(cols)[selectors[0].get_support()])


def cross_validate_repeated(model_name, X, y, repeats, groups, train_ok=None, max_features=None):
    preds = np.zeros((repeats, len(y)))
    baseline = np.zeros((repeats, len(y)))
    eval_ok = np.ones(len(y), dtype=bool) if train_ok is None else train_ok
    fold_r2 = []
    for r in range(repeats):
        for train, test in group_folds(groups, 5, r):
            fit_idx = train if train_ok is None else train[train_ok[train]]
            fold_model = make_model(model_name, X.shape[1], max_features).fit(X[fit_idx], y[fit_idx])
            preds[r, test] = fold_model.predict(X[test])
            baseline[r, test] = y[fit_idx].mean()
            test_eval = test[eval_ok[test]]
            fold_r2.append(r2_score(y[test_eval], preds[r, test_eval]))
    pooled_r2 = np.array([r2_score(y[eval_ok], preds[r][eval_ok]) for r in range(repeats)])
    return preds, baseline, np.array(fold_r2), pooled_r2


def fit_and_cross_validate(df, cols, model_name, repeats, train_ok=None, quiet=False, max_features=None):
    X = df[cols].to_numpy(dtype=float)
    y = df["total_score"].to_numpy(dtype=float)

    fit_rows = np.ones(len(y), dtype=bool) if train_ok is None else train_ok
    model = make_model(model_name, X.shape[1], max_features).fit(X[fit_rows], y[fit_rows])
    r2 = model.score(X[fit_rows], y[fit_rows])
    preds, baseline, fold_r2, pooled_r2 = cross_validate_repeated(model_name, X, y, repeats, df["record_id"].to_numpy(), train_ok, max_features)

    if not quiet:
        n_used = len(cols) if max_features is None else min(len(cols), max_features)
        label = " + ".join(cols) if len(cols) <= 3 else f"{n_used} features"
        print(f"{label} | {model_name} | R2 {r2:.3f} | CV R2 {pooled_r2.mean():.3f} (sd {pooled_r2.std():.3f})")
    return {
        "model": model, "r2": r2, "pooled_r2": pooled_r2, "fold_r2": fold_r2,
        "preds": preds, "baseline": baseline, "cv_pred": preds.mean(axis=0),
    }


def fmt(values, digits=2):
    return f"{values.mean():.{digits}f} {values.std():.{digits}f}"


def safe_corr(preds, y, method):
    values = []
    for p in preds:
        if np.std(p) == 0:
            values.append(np.nan)
        elif method == "pearson":
            values.append(np.corrcoef(p, y)[0, 1])
        else:
            values.append(spearmanr(p, y)[0])
    values = np.array(values)
    return values


def subset_mean(values, mask):
    return values[..., mask].mean() if mask.any() else np.nan


def cv_metrics(y, preds, baseline):
    err = preds - y
    base_err = baseline - y
    return [
        ("MAE", np.abs(err).mean(axis=1), np.abs(base_err).mean(axis=1)),
        ("MSE", (err ** 2).mean(axis=1), (base_err ** 2).mean(axis=1)),
        ("RMSE", np.sqrt((err ** 2).mean(axis=1)), np.sqrt((base_err ** 2).mean(axis=1))),
        ("median AE", np.median(np.abs(err), axis=1), np.median(np.abs(base_err), axis=1)),
        ("bias (pred - true)", err.mean(axis=1), base_err.mean(axis=1)),
        ("pearson r", safe_corr(preds, y, "pearson"), None),
        ("spearman rho", safe_corr(preds, y, "spearman"), None),
    ]


def scatter_by_visit_type(ax, df, x_col, y_col):
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
    ap = argparse.ArgumentParser(description="Predict DSRS total score from session-level speaker:DA group shares.")
    ap.add_argument("--dataset", default="dataset.csv", help="data_loader.py output")
    ap.add_argument("--daseg-dir", required=True, help="run_daseg.py output dir")
    ap.add_argument("--timepoints", nargs="+", default=None, help="timepoints to use (e.g. T0 T2 T3, or 'all')")
    ap.add_argument("--blacklist", default=None,
                    help="file of record ids to exclude, one per line")
    ap.add_argument("--exclude-bilingual", action="store_true",
                    help="drop rows whose transcript is flagged is_bilingual in the dataset")
    ap.add_argument("--train-without-hard", action="store_true",
                    help="train on the visits that are not dyadic or bilingual, and score the dyadic and bilingual visits as a separate held-out set (overrides --exclude-bilingual)")
    ap.add_argument("--auc-cutoff", type=float, default=19, help="DSRS cutoff for the AUC reported next to each arm")
    ap.add_argument("--model", choices=["elasticnet", "ridge", "linear"], default="elasticnet",
                    help="model for the all-features fit")
    ap.add_argument("--max-features", type=int, default=None,
                    help="use at most this many features in each model, chosen inside every training fold by a univariate F test")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeats of 5-fold CV")
    ap.add_argument("--save-model", default=None, help="path to save the all-features model for the app (e.g. models/speaker_da.joblib)")
    ap.add_argument("--output-dir", default=".", help="folder for outputs")
    args = ap.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "da_features_dataset.csv"
    plot_path = output_dir / "da_features_vs_severity.png"
    cv_plot_path = output_dir / "da_features_vs_severity_cv.png"
    prediction_path = output_dir / "da_cv_predictions.csv"

    # Load data
    dataset = pd.read_csv(args.dataset)
    required = ["record_id", "site", "timepoint", "total_score", "transcript_files"]
    if args.exclude_bilingual or args.train_without_hard:
        required.append("is_bilingual")

    missing = [c for c in required if c not in dataset.columns]
    if missing:
        raise SystemExit(f"{args.dataset} is missing column(s) {missing}.")

    print(f"\nLoaded {len(dataset)} row(s) from {args.dataset} (one row per patient per timepoint).")

    # Remove blacklisted IDs (bad diarization or sus DSRS)
    if args.blacklist:
        blacklist_path = Path(args.blacklist)
        if not blacklist_path.is_file():
            raise SystemExit(f"blacklist file not found: {blacklist_path}")

        blacklist = {line.strip().upper() for line in blacklist_path.read_text().splitlines()} - {""}
        ids = dataset["record_id"].astype(str).str.strip().str.upper()
        on_blacklist = ids.isin(blacklist)

        print(f"\nBlacklist: {len(blacklist)} id(s) from {blacklist_path}. Their rows are removed at every timepoint.")
        for rid, n in ids[on_blacklist].value_counts().sort_index().items():
            print(f"excluded: {rid} ({n} row(s))")
        for rid in sorted(blacklist - set(ids)):
            print(f"blacklisted id not found in the dataset: {rid}")

        dataset = dataset[~on_blacklist].reset_index(drop=True)

    if args.exclude_bilingual and not args.train_without_hard:
        # I believe this is effectively code bloat at this point
        bilingual = dataset["is_bilingual"].astype(str).str.lower().eq("true")
        print(f"\nBilingual: {int(bilingual.sum())} row(s) removed because their transcript is flagged bilingual.")
        dataset = dataset[~bilingual].reset_index(drop=True)

    # Grab timepoints
    if args.timepoints and "all" not in [t.lower() for t in args.timepoints]:
        wanted = [t.upper() for t in args.timepoints]
        unknown = sorted(set(wanted) - set(dataset["timepoint"].unique()))
        if unknown:
            raise SystemExit(f"--timepoints {unknown} not in the dataset (found: {sorted(dataset['timepoint'].unique())}).")

        dataset = dataset[dataset["timepoint"].isin(wanted)].reset_index(drop=True)
        print(f"\nTimepoints used: {', '.join(wanted)} ({len(dataset)} row(s)).")
    else:
        print(f"\nTimepoints used: all ({len(dataset)} row(s)).")

    # Match each visit to its daseg output
    print("\nMatching each row to its daseg output. Rows with no usable daseg file are listed and skipped:")
    df = build_da_dataset(dataset, Path(args.daseg_dir))

    print(f"{len(dataset) - len(df)} row(s) skipped, {len(df)} row(s) kept.")
    if df["record_id"].nunique() < 5:
        raise SystemExit(f"Only {df['record_id'].nunique()} patient(s) with daseg output, need at least 5.")

    df.to_csv(dataset_path, index=False)
    print(f"\n{len(df)} rows from {df['record_id'].nunique()} patients; by site: " + ", ".join(f"{k} {v}" for k, v in df["site"].value_counts().items()) +
          "; by timepoint: " + ", ".join(f"{k} {v}" for k, v in sorted(df["timepoint"].value_counts().items())))

    # Identify which visits are dyadic or bilingual
    dyad = df[[f"caregiver_{g}_pct" for g in DA_GROUPS]].sum(axis=1) == 0
    bilingual = df["is_bilingual"].astype(str).str.lower().eq("true")
    hard = (dyad | bilingual).to_numpy()
    df["visit_type"] = np.select([dyad.to_numpy() & bilingual.to_numpy(), dyad.to_numpy(), bilingual.to_numpy()], ["dyadic and bilingual", "dyadic", "bilingual"], default="other")

    train_ok = ~hard if args.train_without_hard else None
    print(f"dyadic visits (no caregiver segments): {int(dyad.sum())}, bilingual visits: {int(bilingual.sum())}" + (", used for scoring only, not for training" if args.train_without_hard else ""))
    eval_mask = train_ok if train_ok is not None else np.ones(len(df), dtype=bool)
    print(f"headline metrics use {int(eval_mask.sum())} visits, and the dyadic and bilingual columns score those visits separately")
    plot_sets = [("", eval_mask)]

    if train_ok is not None and hard.any():
        plot_sets.append(("_hard", hard))

    df["site_is_nyu"] = (df["site"].astype(str).str.lower() == "nyu").astype(float)

    # Features: each speaker's share of their own dialogue acts, dropping any that never vary
    candidates = [c for c in PAIR_COLS if not c.endswith("_other_pct")]
    constant = [c for c in candidates if df[c].std() == 0]
    if constant:
        print("\nDropped features that never vary in this data: " + ", ".join(constant))
    features = [c for c in candidates if c not in constant]

    # Direct fit of each feature against the DSRS, one panel per feature
    truth = df["total_score"].to_numpy(dtype=float)
    panels = {col: fit_and_cross_validate(df, [col], "linear", args.repeats, train_ok, quiet=True) for col in features}
    ncols = 6
    nrows = int(np.ceil(len(features) / ncols))

    for suffix, mask in plot_sets:
        sub = df[mask]
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)
        axes = axes.ravel()

        for ax, col in zip(axes, features):
            panel = panels[col]
            model, r2, pooled_r2 = panel["model"], panel["r2"], panel["pooled_r2"]
            scatter_by_visit_type(ax, sub, col, "total_score")
            x_line = np.linspace(sub[col].min(), sub[col].max(), 100)
            ax.plot(x_line, model.predict(x_line.reshape(-1, 1)), color="firebrick", linewidth=1.5)
            note = f"n={int(mask.sum())}\nMAE={np.abs(panel['cv_pred'][mask] - truth[mask]).mean():.2f}" if suffix else f"R\u00b2={r2:.2f}\nCV R\u00b2={pooled_r2.mean():.2f}"
            ax.text(0.05, 0.95, note, transform=ax.transAxes,
                    ha="left", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
            role, group = col[:-len("_pct")].split("_", 1)
            ax.set_xlabel(f"{role}:{group.replace('_', ' ')} (% of {role}'s DA segments)", fontsize=8)
            ax.set_ylabel("Total DSRS score")
            add_legend(ax, sub, 8, "upper right")

        for ax in axes[len(features):]:
            ax.axis("off")

        fig.tight_layout()
        path = plot_path.with_name(f"{plot_path.stem}{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\nWrote {path}")

    # Fit all features together
    print(f"\nAll-feature model ({args.model}), {len(features)} speaker:DA proportions within each speaker:")
    res = fit_and_cross_validate(df, features, args.model, args.repeats, train_ok, max_features=args.max_features)
    model, r2, pooled_r2, cv_pred = res["model"], res["r2"], res["pooled_r2"], res["cv_pred"]

    coefs = sorted(zip(selected_columns(model, features), model[-1].coef_), key=lambda t: abs(t[1]), reverse=True)
    print("largest standardized coefficients: " + ", ".join(f"{c} {v:+.2f}" for c, v in coefs[:10] if v != 0.0))

    metrics = cv_metrics(df["total_score"].to_numpy(dtype=float)[eval_mask], res["preds"][:, eval_mask], res["baseline"][:, eval_mask])
    print(f"CV MAE {metrics[0][1].mean():.2f} (mean-only {metrics[0][2].mean():.2f}), RMSE {metrics[2][1].mean():.2f}, bias {metrics[4][1].mean():+.2f}, "
          f"MSE reduction vs the mean {1 - metrics[1][1].mean() / metrics[1][2].mean():.3f}")

    y_all = df["total_score"].to_numpy(dtype=float)
    y_eval = y_all[eval_mask]
    if hard.any():
        err = np.abs(res["cv_pred"] - y_all)
        print(f"\nDyadic or bilingual visits ({int(hard.sum())}): MAE {err[hard].mean():.2f}. The rest ({int((~hard).sum())}): MAE {err[~hard].mean():.2f}.")

    # Ablation: same rows, folds and model for each feature set
    joint = "acts (own share), all speakers"
    speaker_only = "speaker only (each speaker's share of all segments)"
    da_only = "dialogue act only (each act's share of all segments)"
    overall_all = "acts (overall share), all speakers"
    overall_cols = [c for c in SHARE_COLS if not c.endswith("_other_of_all_pct")]
    subsets = [("all speakers", ROLES), ("provider only", ("provider",)), ("patient only", ("patient",)), ("caregiver only", ("caregiver",)),
               ("patient + caregiver", ("patient", "caregiver")), ("patient + provider", ("patient", "provider")), ("caregiver + provider", ("caregiver", "provider"))]
    arms = {
        "site only": ["site_is_nyu"],
        speaker_only: [f"{r}_total_pct" for r in ROLES],
        da_only: [f"{g}_total_pct" for g in DA_GROUPS if g != "other"],
    }

    for label, roles in subsets:
        arms[f"acts (own share), {label}"] = [c for c in features if c.startswith(tuple(f"{r}_" for r in roles))]

    for label, roles in subsets:
        arms[f"acts (overall share), {label}"] = [c for c in overall_cols if c.startswith(tuple(f"{r}_" for r in roles))]

    # Look at each arm of comparison, including what share the DAs are of
    arm_results, arm_cols = {}, {}
    dyad_mask, bilingual_mask = dyad.to_numpy(), bilingual.to_numpy()
    print("\nAblation, same rows, folds and model for every arm:")
    for name, cols in arms.items():
        cols = [c for c in cols if df[c].std() > 0]
        if not cols:
            continue
        
        arm = res if name == joint else fit_and_cross_validate(df, cols, args.model, args.repeats, train_ok, quiet=True, max_features=args.max_features)
        arm_results[name] = arm
        arm_cols[name] = cols
        n_used = len(cols) if args.max_features is None else min(len(cols), args.max_features)
        err = np.abs(arm["preds"] - y_all)
        auc = f"{roc_auc_score(y_eval >= args.auc_cutoff, arm['cv_pred'][eval_mask]):.3f}" if 0 < (y_eval >= args.auc_cutoff).sum() < len(y_eval) else "n/a"
        print(f"{name:<54} {n_used:>2} features | CV R2 {arm['pooled_r2'].mean():.3f} | MAE {err[:, eval_mask].mean():.2f} "
              f"(mean-only {np.abs(arm['baseline'] - y_all)[:, eval_mask].mean():.2f}) | AUC {auc} | "
              f"MAE dyadic {subset_mean(err, dyad_mask):.2f}, bilingual {subset_mean(err, bilingual_mask):.2f}, neither {subset_mean(err, ~hard):.2f}")

    overall_model = arm_results[overall_all]["model"]
    overall_coefs = sorted(zip(selected_columns(overall_model, arm_cols[overall_all]), overall_model[-1].coef_), key=lambda t: abs(t[1]), reverse=True)
    print(f"\nacts (overall share), all speakers: in-sample R2 {arm_results[overall_all]['r2']:.3f}, CV R2 {arm_results[overall_all]['pooled_r2'].mean():.3f}")
    print("largest standardized coefficients, overall share: " + ", ".join(f"{c} {v:+.2f}" for c, v in overall_coefs[:10] if v != 0.0))

    # Save results and plotting
    confusion_arms = [da_only, joint, overall_all]
    save_confusions(y_all, df["record_id"].to_numpy(), {name: arm_results[name]["cv_pred"] for name in confusion_arms}, train_ok, hard, output_dir, "da")

    predictions = df[["record_id", "visit", "site", "timepoint", "total_score"]].copy()
    predictions["dyadic"] = dyad.to_numpy()
    predictions["bilingual"] = bilingual.to_numpy()
    for name, arm in arm_results.items():
        predictions[name] = arm["cv_pred"]
    predictions.to_csv(prediction_path, index=False)

    df["cv_predicted"] = cv_pred

    for suffix, mask in plot_sets:
        sub = df[mask]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        scatter_by_visit_type(ax, sub, "total_score", "cv_predicted")
        lo = min(sub["total_score"].min(), sub["cv_predicted"].min())
        hi = max(sub["total_score"].max(), sub["cv_predicted"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
        note = f"n={int(mask.sum())}\nMAE={np.abs(sub['cv_predicted'] - sub['total_score']).mean():.2f}" if suffix else f"R\u00b2={r2:.2f}\nCV R\u00b2={pooled_r2.mean():.2f} \u00b1 {pooled_r2.std():.2f}"
        ax.set_xlabel("True total DSRS score")
        ax.set_ylabel(f"CV predicted total DSRS score (mean of {args.repeats} repeats of 5-fold)\n"
                      f"({args.model}, all speaker:DA group shares)")
        add_legend(ax, sub, 9, "upper left", note)
        fig.tight_layout()
        path = cv_plot_path.with_name(f"{cv_plot_path.stem}{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Wrote {path}")

    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": res["model"], "features": features, "model_name": args.model,
            "cv": {"r2": float(pooled_r2.mean()), "mae": float(metrics[0][1].mean())},
            "n_rows": int(len(df)), "n_patients": int(df["record_id"].nunique()),
            "timepoints": sorted(df["timepoint"].unique().tolist()),
        }, save_path)
        print(f"\nSaved model to {save_path}")

    print(f"\nWrote {dataset_path}, {prediction_path}")


if __name__ == "__main__":
    main()
