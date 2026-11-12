"""
Skyler Heininger
COSC 272
11/10/2026

This script uses dialogue act grouping within a CNN to predict a session's DSRS. 

Considering a CNN is more a sequence to sequence model, we instead use the CNN more for local feature extraction,
and then take the mean and max of each filter in the CNN (32 filters, so 64 total with max and mean). These 64 values
are then put through a dropout and linear layer, whose output is then the prediction for the standardized DSRS score.

We mean-standardize the DSRS scores before training. We split the DSRS into each fold using 5-fold cross-validation,
splitting by patient, since we use multiple visits per-patient.

Within this script, Claude Sonnet 5.5 was used to help with graphing functions, some errors run into with tensor shapes, 
and helping to troubleshoot some issues ran into with the data loading (this changed a couple of times through the project, 
due to varying amounts of data made available, so Sonnet helped to iterate on existing data loading pipelines).
(this is the same across cnn_predict_sequence_da.py, elasticnet_predict_session_level_da.py, and elasticnet_speaker_role_prediction.py,
as there is a bit of repeated code among them)
"""

import argparse
import re
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VISIT_STYLES = {
    "other": ("#4C72B0", "o"),
    "dyadic": ("#DD8452", "s"),
    "bilingual": ("#55A868", "^"),
    "dyadic and bilingual": ("#C44E52", "D"),
}

# Constants for labels
BAND_LABELS = ("0-18", "19-36", "37-54")
BAND_EDGES = (18.49, 36.49)
L1_RATIOS = [0.1, 0.5, 0.9, 1.0]
SPEAKER_ONLY = "speaker shares only (linear)"
CNN = "cnn"
CNN_SPEAKERS = "cnn, dialogue acts hidden"
AUC_CUTOFF = 19

# Training params
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 1e-2
DROPOUT = 0.3
TOKEN_DROPOUT = 0.1
SPEAKER_DIM = 3
GROUP_DIM = 6
FILTERS = 32
KERNEL = 3
LAYERS = 2
SET_SEED = 10

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
N_GROUPS = len(DA_GROUPS)
DROP_ID = N_GROUPS
MIN_SEGMENTS = 10

def map_da_to_group(da):
    return _DA_TO_GROUP.get(str(da).strip(), "other")

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

def load_session(path):
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return None

    # Need the DA's to actually be preocessed already
    if not {"role", "Proc_DA", "DA_number"}.issubset(df.columns):
        return None

    # Need valid DAs
    df = df[df["DA_number"] != ""]
    if df.empty:
        return None

    # Split into DA segments
    segments = df.groupby("DA_number", sort=False).agg(role=("role", "first"), da=("Proc_DA", "last"))
    segments["role"] = segments["role"].str.strip().str.lower()
    segments = segments[segments["role"].isin(ROLES)]

    if len(segments) < MIN_SEGMENTS:
        return None

    return {
        "speaker": np.array([ROLES.index(r) for r in segments["role"]], dtype=np.int64),
        "group": np.array([DA_GROUPS.index(map_da_to_group(d)) for d in segments["da"]], dtype=np.int64),
    }

def build_sessions(dataset, daseg_dir):
    # Grab all sessions from daseg dir
    keep, sessions = [], []

    for i, rec in enumerate(dataset.itertuples(index=False)):
        found = None

        for name in str(rec.transcript_files).split(";"):
            name = name.strip()
            if not name:
                continue

            found = load_session(daseg_dir / f"{Path(name).stem}_with_predictions.csv")
            if found is not None:
                break

        if found is None:
            print(f"no usable daseg output: {rec.record_id} {rec.timepoint} {rec.transcript_files}")
            continue

        keep.append(i)
        sessions.append(found)

    return dataset.iloc[keep].reset_index(drop=True), sessions


def apply_blacklist(dataset, blacklist_path):
    # Some IDs we do not trust the DSRS of, or has poor diarization
    if not blacklist_path.is_file():
        raise SystemExit(f"blacklist file not found: {blacklist_path}")

    blacklist = {normalize_id(line) for line in blacklist_path.read_text().splitlines()} - {""}
    ids = dataset["record_id"].map(normalize_id)
    on_blacklist = ids.isin(blacklist)
    print(f"Blacklist: {len(blacklist)} ids from {blacklist_path}")

    for rid, n in ids[on_blacklist].value_counts().sort_index().items():
        print(f"excluded: {rid} ({n} rows)")

    for rid in sorted(blacklist - set(ids)):
        print(f"not in dataset: {rid}")

    return dataset[~on_blacklist].reset_index(drop=True)


def load_dataset(args):
    # Load dataset obect, contains a lot of pre-processed info
    dataset = pd.read_csv(args.dataset)
    required = ["record_id", "site", "timepoint", "total_score", "transcript_files"]

    if args.train_without_hard:
        required.append("is_bilingual")

    missing = [c for c in required if c not in dataset.columns]
    if missing:
        raise SystemExit(f"{args.dataset} is missing column(s) {missing}.")

    print(f"Loaded {len(dataset)} rows from {args.dataset}")

    if args.blacklist:
        dataset = apply_blacklist(dataset, Path(args.blacklist))

    if args.timepoints and "all" not in [t.lower() for t in args.timepoints]:
        wanted = [t.upper() for t in args.timepoints]
        unknown = sorted(set(wanted) - set(dataset["timepoint"].unique()))

        if unknown:
            raise SystemExit(f"--timepoints {unknown} not in the dataset (found: {sorted(dataset['timepoint'].unique())}).")

        dataset = dataset[dataset["timepoint"].isin(wanted)].reset_index(drop=True)

    print(f"Timepoints kept: {len(dataset)} rows")
    return dataset


def flag_hard_sessions(df, sessions):
    # We consider bilingual or dyadic sessions as hard. Note that sessions that just contain the 
    # caregiver and clinician are considered triadic, since the patient had to be in the care visits.
    # So the only true dyadic sessions are one where there is no caregiver turns.

    caregiver = ROLES.index("caregiver")
    dyad = np.array([not (s["speaker"] == caregiver).any() for s in sessions])

    if "is_bilingual" in df.columns:
        bilingual = df["is_bilingual"].astype(str).str.lower().eq("true").to_numpy()
    else:
        bilingual = np.zeros(len(df), dtype=bool)

    return dyad, bilingual


class SessionCNN(nn.Module):
    def __init__(self, speaker_dim=SPEAKER_DIM, group_dim=GROUP_DIM, n_filters=FILTERS,
                 kernel=KERNEL, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.speaker = nn.Embedding(len(ROLES), speaker_dim)
        self.group = nn.Embedding(N_GROUPS + 1, group_dim)
        convs, dim = [], speaker_dim + group_dim

        for _ in range(layers):
            convs.append(nn.Conv1d(dim, n_filters, kernel, padding=kernel // 2))
            dim = n_filters

        self.convs = nn.ModuleList(convs)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * n_filters, 1)

    def forward(self, speaker, group, mask):
        x = torch.cat([self.speaker(speaker), self.group(group)], dim=2)
        maskf = mask.float()
        h = x.transpose(1, 2)

        for conv in self.convs:
            h = torch.relu(conv(h)) * maskf[:, None, :]

        # Grab max and mean for each filter
        top = h.masked_fill(~mask[:, None, :], float("-inf")).amax(dim=2)
        mean = h.sum(dim=2) / maskf.sum(dim=1, keepdim=True)

        return self.head(self.drop(torch.cat([top, mean], dim=1))).squeeze(1)


def batch_arrays(sessions, idx, drop_p, rng, hide_groups):
    # Batching for the folds
    longest = max(len(sessions[i]["speaker"]) for i in idx)
    speaker = np.zeros((len(idx), longest), dtype=np.int64)
    group = np.full((len(idx), longest), DROP_ID, dtype=np.int64)
    mask = np.zeros((len(idx), longest), dtype=bool)

    for row, i in enumerate(idx):
        size = len(sessions[i]["speaker"])
        speaker[row, :size] = sessions[i]["speaker"]
        group[row, :size] = sessions[i]["group"]
        mask[row, :size] = True

    if hide_groups:
        group[:] = DROP_ID
    elif drop_p > 0:
        group[mask & (rng.random(group.shape) < drop_p)] = DROP_ID

    return speaker, group, mask


def to_tensors(batch):
    return tuple(torch.from_numpy(part) for part in batch)


def balanced_weights(y):
    # Weight each visit by the inverse frequency of its severity band, scaled so the average weight is 1
    bands = to_band(y)
    weights = 1.0 / np.bincount(bands, minlength=len(BAND_LABELS))[bands]
    return (weights / weights.mean()).astype(np.float32)


def train_net(sessions, y, train, seed, hide_groups, class_weights=False):
    # Training loop. Should be pretty self-explanatory. Using MSE for loss
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, std = float(y[train].mean()), max(float(y[train].std()), 1e-6)
    target = ((y[train] - mean) / std).astype(np.float32)

    # Every visit counts equally unless class weighting is on
    weights = balanced_weights(y[train]) if class_weights else np.ones(len(train), dtype=np.float32)

    net = SessionCNN()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = int(np.ceil(len(train) / BATCH_SIZE))
    # Reduce LR throughout, gets pretty sensitive to LR throughout
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * steps_per_epoch)
    net.train()

    for _ in range(EPOCHS):
        order = rng.permutation(len(train))

        for start in range(0, len(train), BATCH_SIZE):
            pick = order[start:start + BATCH_SIZE]
            batch = to_tensors(batch_arrays(sessions, train[pick], TOKEN_DROPOUT, rng, hide_groups))

            # MSE, with each visit scaled by its weight
            loss = (((net(*batch) - torch.from_numpy(target[pick])) ** 2) * torch.from_numpy(weights[pick])).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

    return net, mean, std


def predict_net(net, sessions, test, mean, std, hide_groups):
    # Forward pass for prediction
    net.eval()
    rng = np.random.default_rng(0)
    outputs = []

    with torch.no_grad():
        for start in range(0, len(test), 16):
            batch = to_tensors(batch_arrays(sessions, test[start:start + 16], 0.0, rng, hide_groups))
            outputs.append(net(*batch).numpy())

    return np.concatenate(outputs) * std + mean


def save_net(path, net, mean, std, class_weights):
    # Everything the app needs to rebuild the network and turn its output back into DSRS points
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "mean": float(mean),
        "std": float(std),
        "config": {"speaker_dim": SPEAKER_DIM, "group_dim": GROUP_DIM, "n_filters": FILTERS, "kernel": KERNEL, "layers": LAYERS, "dropout": DROPOUT},
        "roles": list(ROLES),
        "da_groups": list(DA_GROUPS),
        "da_to_group": dict(_DA_TO_GROUP),
        "class_weights": bool(class_weights),
    }, path)


def cross_validate_neural(name, sessions, y, groups, hide_groups, train_ok=None, class_weights=False):
    # Runs the cross validation. Note that we made some pieces for repeating, with different seeds, to help
    # validate results. 

    # The shapes of these look a little odd, this took a couple of different forms over time, this modification
    # was the simplest to do from an earlier version.
    preds = np.zeros((1, len(y)))
    baseline = np.zeros((1, len(y)))

    for k, (train, test) in enumerate(group_folds(groups, 5, 0)):
        if train_ok is not None:
            train = train[train_ok[train]]

        began = time.time()
        net, mean, std = train_net(sessions, y, train, SET_SEED, hide_groups, class_weights)
        preds[0, test] = predict_net(net, sessions, test, mean, std, hide_groups)
        baseline[0, test] = y[train].mean()
        print(f"{name}: fold {k + 1} of 5, {time.time() - began:.0f}s")

    return preds, baseline


def make_speaker_pipeline():
    return make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=L1_RATIOS, alphas=50, max_iter=20000, eps=1e-2),
    )


def speaker_shares(sessions):
    shares = np.zeros((len(sessions), len(ROLES)))

    for i, s in enumerate(sessions):
        shares[i] = np.bincount(s["speaker"], minlength=len(ROLES)) / len(s["speaker"])

    return shares


def cross_validate_linear(name, make_model, X, y, groups, train_ok=None):
    preds = np.zeros((1, len(y)))
    baseline = np.zeros((1, len(y)))

    for train, test in group_folds(groups, 5, 0):
        if train_ok is not None:
            train = train[train_ok[train]]

        preds[0, test] = make_model().fit(X[train], y[train]).predict(X[test])
        baseline[0, test] = y[train].mean()

    print(f"{name}: done")

    return preds, baseline

def spearman(p, y):
    return np.nan if np.std(p) == 0 else float(spearmanr(p, y)[0])


def slope(p, y):
    return np.polyfit(p, y, 1)[0] if np.std(p) > 0 else np.nan


def metrics(preds, baseline, y):
    err, base_err = preds - y, baseline - y
    mse, base_mse = (err ** 2).mean(axis=1), (base_err ** 2).mean(axis=1)

    return {
        "r2": np.array([r2_score(y, p) for p in preds]),
        "mae": np.abs(err).mean(axis=1),
        "base_mae": np.abs(base_err).mean(axis=1),
        "rmse": np.sqrt(mse),
        "spearman": np.array([spearman(p, y) for p in preds]),
        "slope": np.array([slope(p, y) for p in preds]),
        "mse_reduction": 1 - mse / base_mse,
    }


def mean_sd(values, digits=3):
    if np.all(np.isnan(values)):
        return "n/a"

    return f"{np.nanmean(values):.{digits}f} (sd {np.nanstd(values):.{digits}f})"


def subset_mean(values, mask):
    return values[..., mask].mean() if mask.any() else np.nan


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


def band_lines(y, pred):
    # Each band of DSRS severity
    lines = []
    bands = (("0-18", y <= 18), ("19-36", (y > 18) & (y <= 36)), ("37+", y > 36))
    for label, mask in bands:
        if mask.any():
            lines.append(f"  {label}: n {int(mask.sum())}, MAE {np.abs(pred[mask] - y[mask]).mean():.2f}, bias {(pred[mask] - y[mask]).mean():+.2f}")

    return lines


def plot_predictions(df, y, mean_pred, held_out, path):
    # Genera plotting
    names = list(mean_pred)
    ncols = min(4, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.6 * nrows), squeeze=False)
    lo = min(y.min(), min(p.min() for p in mean_pred.values()))
    hi = max(y.max(), max(p.max() for p in mean_pred.values()))

    for ax in axes.ravel()[len(names):]:
        ax.axis("off")

    for ax, name in zip(axes.ravel(), names):
        pred = mean_pred[name]
        scatter_by_visit_type(ax, df.assign(cv_predicted=pred), "total_score", "cv_predicted")
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
        note = f"n={len(y)}\nMAE={np.abs(pred - y).mean():.1f}" if held_out else f"CV R\u00b2={r2_score(y, pred):.2f}\nMAE={np.abs(pred - y).mean():.1f}"
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("True total DSRS score")
        ax.set_ylabel("CV predicted total DSRS score")
        add_legend(ax, df, 8, "upper left", note)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    ap = argparse.ArgumentParser(description="Predict DSRS total score from the speaker:DA sequence of a session with a small CNN.")
    ap.add_argument("--dataset", default="dataset.csv", help="data_loader.py output")
    ap.add_argument("--daseg-dir", required=True, help="run_daseg.py output dir")
    ap.add_argument("--timepoints", nargs="+", default=None, help="timepoints to use (e.g. T0 T2 T3, or 'all')")
    ap.add_argument("--blacklist", default=None, help="file of record ids to exclude, one per line")
    ap.add_argument("--train-without-hard", action="store_true", help="train on the sessions that are not dyadic or bilingual, and score the dyadic and bilingual sessions as a separate held-out set")
    ap.add_argument("--class-weights", action="store_true", help="weight each training visit's loss by the inverse frequency of its severity band (0-18, 19-36, 37-54) in the training folds, for the CNN arms only")
    ap.add_argument("--save-model", default=None, help="path to save a CNN trained on every usable session, for the app (e.g. models/cnn_da.pth)")
    ap.add_argument("--output-dir", default=".", help="folder for outputs")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    # Load data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "neural_cv_predictions.csv"

    dataset = load_dataset(args)

    print("Loading daseg output:")
    df, sessions = build_sessions(dataset, Path(args.daseg_dir))
    print(f"{len(dataset) - len(df)} skipped, {len(df)} kept")

    if df["record_id"].nunique() < 10:
        raise SystemExit(f"Only {df['record_id'].nunique()} patient(s) with usable sequences, need at least 10.")

    # Identify which sessions are dyadic or bilingual
    dyad, bilingual = flag_hard_sessions(df, sessions)
    hard = dyad | bilingual
    df["visit_type"] = np.select([dyad & bilingual, dyad, bilingual], ["dyadic and bilingual", "dyadic", "bilingual"], default="other")

    # prep data
    y = df["total_score"].to_numpy(dtype=float)
    groups = df["record_id"].to_numpy()
    train_ok = ~hard if args.train_without_hard else None
    eval_mask = train_ok if train_ok is not None else np.ones(len(y), dtype=bool)
    y_eval, groups_eval = y[eval_mask], groups[eval_mask]
    lengths = np.array([len(s["speaker"]) for s in sessions])
    print(f"{len(df)} sessions from {df['record_id'].nunique()} patients, segments per session: min {lengths.min()}, median {int(np.median(lengths))}, max {lengths.max()}")
    print(f"dyadic sessions: {int(dyad.sum())}, bilingual sessions: {int(bilingual.sum())}")
    print(f"headline metrics use {int(eval_mask.sum())} sessions, and the dyadic and bilingual columns score those sessions separately")
    print(f"class weighting by severity band: {args.class_weights}")

    # CV
    print("Running cross-validation:")
    results = {}
    preds, baseline = cross_validate_linear("speaker shares", make_speaker_pipeline, speaker_shares(sessions), y, groups, train_ok)
    results[SPEAKER_ONLY] = {"preds": preds, "baseline": baseline}

    # Train with and without dialogue acts, to see how much DAs add to the model above speaker roles
    n_params = sum(p.numel() for p in SessionCNN().parameters())
    print(f"cnn: {n_params} parameters")
    preds, baseline = cross_validate_neural("cnn", sessions, y, groups, False, train_ok, args.class_weights)
    results[CNN] = {"preds": preds, "baseline": baseline}
    preds, baseline = cross_validate_neural("cnn, dialogue acts hidden", sessions, y, groups, True, train_ok, args.class_weights)
    results[CNN_SPEAKERS] = {"preds": preds, "baseline": baseline}

    # Calculate metrics
    for res in results.values():
        res["stats"] = metrics(res["preds"][:, eval_mask], res["baseline"][:, eval_mask], y_eval)

    # Printing results and saving everything. Most of the logic here is due to differences between
    # saving for running with and without training using bilingual/dyadic visits (since we use them as a held-out
    # test set if we don't explicitly train on them)
    print("Results:")
    for name, res in results.items():
        s = res["stats"]
        print(f"{name} | R2 {mean_sd(s['r2'])} | MAE {mean_sd(s['mae'], 2)} (mean-only {np.mean(s['base_mae']):.2f}) | RMSE {mean_sd(s['rmse'], 2)} | "
              f"Spearman {mean_sd(s['spearman'])} | MSE reduction {mean_sd(s['mse_reduction'])} | slope {mean_sd(s['slope'], 2)}")

    mean_pred = {n: r["preds"].mean(axis=0) for n, r in results.items()}
    mean_base = {n: r["baseline"].mean(axis=0) for n, r in results.items()}
    eval_pred = {n: p[eval_mask] for n, p in mean_pred.items()}
    eval_base = {n: b[eval_mask] for n, b in mean_base.items()}
    names = list(results)

    print(f"AUC for DSRS >= {AUC_CUTOFF}:")
    for name in names:
        print(f"{name}: {roc_auc_score(y_eval >= AUC_CUTOFF, eval_pred[name]):.3f}")

    print("MAE on dyadic sessions, bilingual sessions, and neither:")
    for name in names:
        err = np.abs(mean_pred[name] - y)
        print(f"{name}: {subset_mean(err, dyad):.2f}, {subset_mean(err, bilingual):.2f}, {subset_mean(err, ~hard):.2f}")

    print("Error by score band:")
    for name in names:
        print(f"{name}:\n" + "\n".join(band_lines(y_eval, eval_pred[name])))

    save_confusions(y, groups, {CNN: mean_pred[CNN], CNN_SPEAKERS: mean_pred[CNN_SPEAKERS]}, train_ok, hard, output_dir, "neural")

    predictions = df[[c for c in ("record_id", "visit", "timepoint", "site") if c in df.columns]].copy()
    predictions["total_score"] = y
    predictions["dyadic"] = dyad
    predictions["bilingual"] = bilingual

    for name in names:
        predictions[name] = mean_pred[name]

    predictions.to_csv(predictions_path, index=False)
    plot_sets = [("", eval_mask)]

    if train_ok is not None and hard.any():
        plot_sets.append(("_hard", hard))

    for suffix, mask in plot_sets:
        path = output_dir / f"neural_models_cv{suffix}.png"
        plot_predictions(df[mask], y[mask], {n: p[mask] for n, p in mean_pred.items()}, bool(suffix), path)
        print(f"Wrote {path}")

    print(f"Wrote {predictions_path}")

    # The fold networks each only see part of the data, so train one CNN on every usable session for the app
    if args.save_model:
        final_train = np.where(train_ok)[0] if train_ok is not None else np.arange(len(y))
        began = time.time()
        net, mean, std = train_net(sessions, y, final_train, SET_SEED, False, args.class_weights)
        save_net(Path(args.save_model), net, mean, std, args.class_weights)
        print(f"Saved the CNN, trained on {len(final_train)} sessions ({time.time() - began:.0f}s), to {args.save_model}")

if __name__ == "__main__":
    main()
