"""
Forrest Leung
11/2/26

Predict DSRS total_score from the embeddings track's feature tables using Ridge, Elastic Net, and 1D-CNN models.

To run:
python embeddings/predict_dsrs.py "$DATA"
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

from load_data import EMBED_MODEL, ROLE_CHANNELS
from load_data import get_turn_embeddings, key as visit_key
from load_data import load_dataset, load_roles

RIDGE_ALPHAS = np.logspace(-2, 3, 12)
EN_ALPHAS = np.logspace(-3, 1, 8)
EN_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
INNER_FOLDS = 4
SEED = 0

CNN_HIDDEN = 32
CNN_DROPOUT = 0.3
CNN_WEIGHT_DECAY = 1e-3
CNN_LR = 1e-3
CNN_MAX_EPOCHS = 200
CNN_PATIENCE = 15

# Shared helpers
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    ss_res, ss_tot = float(np.sum(err ** 2)), float(np.sum((y_true - y_true.mean()) ** 2))
    rho = spearmanr(y_true, y_pred).correlation if len(y_true) > 1 else np.nan
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "spearman_rho": rho,
        "n": len(y_true),
    }

def load_feature_table(turn_dynamics_path, topic_duration_path=None, embedding_similarity_path=None,
                       reply_relevance_path=None):
    df = pd.read_csv(turn_dynamics_path)
    if topic_duration_path:
        td = pd.read_csv(topic_duration_path).drop(columns=["path"], errors="ignore")
        df = df.merge(td, on="visit_id", how="left", suffixes=("", "_topic"))
    if embedding_similarity_path:
        es = pd.read_csv(embedding_similarity_path)
        df = df.merge(es, on="visit_id", how="left", suffixes=("", "_embed"))
    if reply_relevance_path:
        rr = pd.read_csv(reply_relevance_path).drop(columns=["embed_model"], errors="ignore")
        df = df.merge(rr, on="visit_id", how="left", suffixes=("", "_reply"))
    return df


EXCLUDE_COLUMNS = {"visit_id", "record_id_clean", "fold", "site", "timepoint",
                   "total_score", "severity_category"}

PROBE_PREFIXES = {"orig": ("probe_orig_",), "neutral": ("probe_neutral_",),
                  "both": ("probe_orig_", "probe_neutral_")}

# Feature groups for different types of features in the dataset
FEATURE_GROUPS = {
    "turn_dynamics": lambda c: not c.startswith("merged_") and (
        c in {"n_turns", "topic_changes_per_turn", "n_patient_responses",
             "patient_topic_shift_rate", "has_coordinator", "has_other"}
        or c.endswith(("_turn_share", "_mean_turn_length", "_topic_changes", "_topic_change_rate"))),
    "topic_duration": lambda c: not c.startswith("merged_") and (
        c == "n_topics" or c.startswith("topic_run_")
        or c.endswith(("_topic_turns_mean", "_topic_words_mean"))),
    # Merged LLM boundaries (from the post-processing step in turn_dynamics.py)
    "llm_boundaries_merged": lambda c: c.startswith("merged_"),
    # Talk-time only: who holds the floor. The baseline every other group should beat (H2 "beyond talk-time")
    "talk_time": lambda c: not c.startswith("merged_") and c.endswith("_turn_share"),
    # H3 reply relevance (reply_relevance.py); its n_reply_* counts are deliberately excluded
    "reply_relevance": lambda c: c.startswith("reply_"),
}

def embedding_similarity_predicate(probe_set):
    prefixes = PROBE_PREFIXES[probe_set]
    return lambda c: c.startswith(prefixes) or c in {"patient_caregiver_alignment", "patient_coherence"}

# "all" = every group; reply_relevance columns only exist when --reply-relevance is passed
ALL_GROUPS = "turn_dynamics+topic_duration+embedding_similarity+reply_relevance"

# Narrower slices of the embedding_similarity features (embedding_similarity.py), by column prefix:
#   probes_max  = the original max-over-turns probe scores only (embedding_similarity minus alignment/coherence)
#   probes_rate = share of turns hitting each probe (length-controlled)
#   probes_role = share of each speaker's turns hitting any probe (who raises the symptom topics)
PROBE_GROUP_PREFIXES = {"probes_max": "probe_", "probes_rate": "prate_", "probes_role": "prole_"}
ALIGNMENT_FEATURES = {"patient_caregiver_alignment", "patient_coherence"}

def group_predicate(group, probe_set="orig"):
    """A group name, or several joined by '+' (e.g. 'talk_time+reply_relevance')."""
    if "+" in group:
        preds = [group_predicate(g, probe_set) for g in group.split("+")]
        return lambda c: any(p(c) for p in preds)
    if group == "all":
        return group_predicate(ALL_GROUPS, probe_set)
    if group == "embedding_similarity":
        return embedding_similarity_predicate(probe_set)
    if group in PROBE_GROUP_PREFIXES:
        sets = ("orig", "neutral") if probe_set == "both" else (probe_set,)
        prefixes = tuple(f"{PROBE_GROUP_PREFIXES[group]}{s}_" for s in sets)
        return lambda c: c.startswith(prefixes)
    if group == "alignment":
        return lambda c: c in ALIGNMENT_FEATURES
    return FEATURE_GROUPS[group]

# Feature selection based on group and probe set
def feature_columns(df, group="all", probe_set="orig"):
    predicate = group_predicate(group, probe_set)
    return [c for c in df.columns
            if c not in EXCLUDE_COLUMNS and pd.api.types.is_numeric_dtype(df[c]) and predicate(c)]

def inner_splits(train_df, k=INNER_FOLDS):
    k = max(2, min(k, train_df["record_id_clean"].nunique()))
    return list(GroupKFold(k).split(train_df, groups=train_df["record_id_clean"]))


def record_predictions(pred_rows, test, pred, model_name, fold):
    for vid, y_true, y_pred in zip(test["visit_id"], test["total_score"], pred):
        pred_rows.append({"visit_id": vid, "fold": int(fold), "model": model_name,
                          "y_true": float(y_true), "y_pred": float(y_pred)})

def run_baseline(df):
    rows, pred_rows = [], []
    for f in sorted(df["fold"].dropna().unique()):
        train, test = df[df["fold"] != f], df[df["fold"] == f]
        pred = np.full(len(test), train["total_score"].mean())
        rows.append({**metrics(test["total_score"], pred), "fold": int(f), "model": "baseline_mean"})
        record_predictions(pred_rows, test, pred, "baseline_mean", f)
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)

# Run linear models (Ridge and Elastic Net) with nested patient-grouped cross-validation
def run_linear(df, feats, model_name):
    rows, coef_rows, pred_rows = [], [], []
    for f in sorted(df["fold"].dropna().unique()):
        train = df[df["fold"] != f].reset_index(drop=True)
        test = df[df["fold"] == f].reset_index(drop=True)
        medians = train[feats].median()
        Xtr_raw, Xte_raw = train[feats].fillna(medians), test[feats].fillna(medians)
        ytr = train["total_score"].to_numpy(float)

        scaler = StandardScaler().fit(Xtr_raw)
        Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

        if model_name == "ridge":
            grid, base = {"alpha": RIDGE_ALPHAS}, Ridge()
        else:
            grid, base = {"alpha": EN_ALPHAS, "l1_ratio": EN_L1_RATIOS}, ElasticNet(max_iter=20000)
        search = GridSearchCV(base, grid, cv=inner_splits(train), scoring="neg_mean_absolute_error")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            search.fit(Xtr, ytr)
        model = search.best_estimator_
        pred = model.predict(Xte)

        rows.append({**metrics(test["total_score"], pred), "fold": int(f), "model": model_name,
                    "best_params": str(search.best_params_)})
        record_predictions(pred_rows, test, pred, model_name, f)
        if model_name == "elastic_net":
            for name, c in sorted(zip(feats, model.coef_), key=lambda x: -abs(x[1])):
                if c != 0:
                    coef_rows.append({"fold": int(f), "feature": name, "coef": float(c)})
    return pd.DataFrame(rows), pd.DataFrame(coef_rows), pd.DataFrame(pred_rows)

# Run ablation studies for Ridge regression with different feature groups
def run_ablations(df, probe_set="orig", groups=("talk_time", "turn_dynamics", "topic_duration",
                                                "embedding_similarity", "llm_boundaries_merged",
                                                "reply_relevance", "talk_time+reply_relevance", "all")):
    rows = []
    for g in groups:
        feats = [c for c in feature_columns(df, g, probe_set) if not df[c].isna().all()]
        if not feats:
            print(f"  ablation group '{g}': no (non-all-NaN) features present, skipping")
            continue
        fold_metrics, _, _ = run_linear(df, feats, "ridge")
        fold_metrics["feature_group"] = g
        fold_metrics["n_features"] = len(feats)
        rows.append(fold_metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# Build sequences of turn embeddings for each visit
def build_sequences(df, roles, cache_dir, model):
    """{visit_id: (T, D+R) float32 array}; D = embedding dim, R = len(ROLE_CHANNELS) role one-hot columns."""
    seqs = {}
    for i, row in df.iterrows():
        emb, role = get_turn_embeddings(row["path"], roles, cache_dir, model)
        seqs[row["visit_id"]] = np.concatenate([emb, role], axis=1) if len(emb) else emb
        print(f"\r  turn embeddings {i + 1}/{len(df)}", end="", flush=True)
    print()
    return seqs

# Collate sequences into a batch for CNN input
def collate(visit_ids, seqs, device):
    import torch
    arrs = [seqs[v] for v in visit_ids]
    dim = next(a.shape[1] for a in arrs if len(a))
    maxlen = max((len(a) for a in arrs), default=1) or 1
    X = np.zeros((len(arrs), maxlen, dim), dtype=np.float32)
    mask = np.zeros((len(arrs), maxlen), dtype=np.float32)
    for i, a in enumerate(arrs):
        if len(a):
            X[i, :len(a)] = a
            mask[i, :len(a)] = 1.0
    return torch.tensor(X, device=device), torch.tensor(mask, device=device)

# Define the CNN model for topic embeddings
def make_cnn(in_dim):
    import torch.nn as nn

    class TopicCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(in_dim, CNN_HIDDEN, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(CNN_HIDDEN, CNN_HIDDEN, kernel_size=5, padding=2)
            self.drop = nn.Dropout(CNN_DROPOUT)
            self.head = nn.Sequential(
                nn.Linear(CNN_HIDDEN, CNN_HIDDEN // 2), nn.ReLU(),
                nn.Dropout(CNN_DROPOUT), nn.Linear(CNN_HIDDEN // 2, 1),
            )

        def forward(self, x, mask):
            h = torch.relu(self.conv1(x.transpose(1, 2)))
            h = self.drop(h)
            h = torch.relu(self.conv2(h)).transpose(1, 2)   # [B, T, hidden]
            m = mask.unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return self.head(pooled).squeeze(-1)

    import torch
    return TopicCNN()

# Run the CNN model with nested patient-grouped cross-validation
def run_cnn(df, seqs, seed=SEED):
    import torch

    device = torch.device("cpu")
    in_dim = next(a.shape[1] for a in seqs.values() if len(a))
    rows, curve_rows, pred_rows = [], [], []
    for f in sorted(df["fold"].dropna().unique()):
        train_all = df[df["fold"] != f]
        test = df[df["fold"] == f]

        # inner patient-grouped split off the outer-training patients, for early stopping
        patients = np.array(train_all["record_id_clean"].unique(), dtype=object)
        rng = np.random.default_rng(seed + int(f))
        rng.shuffle(patients)
        n_val = max(1, int(0.15 * len(patients)))
        val_patients = set(patients[:n_val])
        tr = train_all[~train_all["record_id_clean"].isin(val_patients)]
        va = train_all[train_all["record_id_clean"].isin(val_patients)]

        y_mean, y_std = tr["total_score"].mean(), tr["total_score"].std() or 1.0

        torch.manual_seed(seed + int(f))
        model = make_cnn(in_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY)

        Xtr, Mtr = collate(tr["visit_id"].tolist(), seqs, device)
        ytr = torch.tensor(((tr["total_score"] - y_mean) / y_std).to_numpy(np.float32), device=device)
        Xva, Mva = collate(va["visit_id"].tolist(), seqs, device)
        yva = va["total_score"].to_numpy(float)

        best_val_mae, best_state, bad_epochs = np.inf, None, 0
        for epoch in range(CNN_MAX_EPOCHS):
            model.train()
            opt.zero_grad()
            pred = model(Xtr, Mtr)
            loss = torch.nn.functional.mse_loss(pred, ytr)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(Xva, Mva).cpu().numpy() * y_std + y_mean
            val_mae = float(np.mean(np.abs(val_pred - yva))) if len(yva) else float(loss.item())
            curve_rows.append({"fold": int(f), "epoch": epoch, "train_mse_loss": float(loss.item()),
                               "val_mae": val_mae})
            if val_mae < best_val_mae - 1e-4:
                best_val_mae, best_state, bad_epochs = val_mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad_epochs += 1
                if bad_epochs >= CNN_PATIENCE:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        Xte, Mte = collate(test["visit_id"].tolist(), seqs, device)
        with torch.no_grad():
            pred = model(Xte, Mte).cpu().numpy() * y_std + y_mean
        rows.append({**metrics(test["total_score"], pred), "fold": int(f), "model": "cnn",
                    "best_val_mae": round(best_val_mae, 3), "epochs_trained": epoch + 1,
                    "early_stopped": (epoch + 1) < CNN_MAX_EPOCHS})
        record_predictions(pred_rows, test, pred, "cnn", f)
    return pd.DataFrame(rows), pd.DataFrame(curve_rows), pd.DataFrame(pred_rows)

# Write a summary of the prediction results to a markdown file
def write_summary(results_dir, all_results, coefs, ablations=None):
    agg = all_results.groupby("model")[["mae", "rmse", "r2", "spearman_rho"]].agg(["mean", "std"])
    lines = ["# DSRS prediction results (embeddings track)\n",
             "One row per fold, patient-grouped folds shared across all models. "
             "Mean ± std across folds:\n", "| model | MAE | RMSE | R2 | Spearman rho |",
             "|---|---|---|---|---|"]
    for model in agg.index:
        m = agg.loc[model]
        lines.append(f"| {model} | {m['mae']['mean']:.2f} ± {m['mae']['std']:.2f} | "
                     f"{m['rmse']['mean']:.2f} ± {m['rmse']['std']:.2f} | "
                     f"{m['r2']['mean']:.2f} ± {m['r2']['std']:.2f} | "
                     f"{m['spearman_rho']['mean']:+.2f} ± {m['spearman_rho']['std']:.2f} |")
    if ablations is not None and len(ablations):
        aagg = ablations.groupby("feature_group")[["mae", "rmse", "r2", "spearman_rho"]].agg(["mean", "std"])
        n_feats = ablations.groupby("feature_group")["n_features"].first()
        lines += ["\n## Ridge ablations by feature group\n", "| feature group | n features | MAE | RMSE | R2 | Spearman rho |",
                 "|---|---|---|---|---|---|"]
        for g in aagg.index:
            m = aagg.loc[g]
            lines.append(f"| {g} | {n_feats[g]} | {m['mae']['mean']:.2f} ± {m['mae']['std']:.2f} | "
                         f"{m['rmse']['mean']:.2f} ± {m['rmse']['std']:.2f} | "
                         f"{m['r2']['mean']:.2f} ± {m['r2']['std']:.2f} | "
                         f"{m['spearman_rho']['mean']:+.2f} ± {m['spearman_rho']['std']:.2f} |")
    if len(coefs):
        freq = coefs.groupby("feature")["fold"].nunique().sort_values(ascending=False)
        lines += ["\n## Elastic Net: features with non-zero coefficients (folds kept in / total folds)\n"]
        for feat, n in freq.items():
            lines.append(f"- {feat}: {n}/{coefs['fold'].nunique()}")
    path = os.path.join(results_dir, "summary.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--turn-dynamics", help="defaults to <data_dir>/derived/turn_dynamics_features.csv")
    ap.add_argument("--topic-duration", help="optional extra topic-duration feature CSV, left-joined on "
                    "visit_id (turn_dynamics_features.csv already includes these by default)")
    ap.add_argument("--embedding-similarity", help="embedding_similarity.py output, left-joined on visit_id; "
                    "defaults to <data_dir>/derived/embedding_similarity_features.csv if present")
    ap.add_argument("--reply-relevance", help="reply_relevance.py output, left-joined on visit_id (H3 features); "
                    "not joined by default, so 'all' only changes when you pass it")
    ap.add_argument("--probe-set", choices=["orig", "neutral", "both"], default="orig",
                    help="which probe wording enters the model (default: %(default)s); "
                         "the two sets are never mixed unless 'both' is chosen explicitly")
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--embeddings-cache", default="outputs/turn_embeddings")
    ap.add_argument("--skip-cnn", action="store_true", help="skip the CNN (e.g. no torch, or just comparing linear models)")
    ap.add_argument("--ablations", action="store_true", help="also run the Ridge feature-group ablations")
    args = ap.parse_args()

    turn_dynamics_path = args.turn_dynamics or os.path.join(args.data_dir, "derived", "turn_dynamics_features.csv")
    embed_sim_path = args.embedding_similarity or os.path.join(args.data_dir, "derived", "embedding_similarity_features.csv")
    if not os.path.exists(embed_sim_path):
        embed_sim_path = None
    df = load_feature_table(turn_dynamics_path, args.topic_duration, embed_sim_path, args.reply_relevance)
    df = df.dropna(subset=["total_score", "fold"]).reset_index(drop=True)
    feats = feature_columns(df, "all", args.probe_set)
    all_nan = [c for c in feats if df[c].isna().all()]
    if all_nan:
        print(f"  dropping {len(all_nan)} all-NaN feature(s) (median imputation can't help these): {all_nan}")
        feats = [c for c in feats if c not in all_nan]
    print(f"{len(df)} visits, {df['record_id_clean'].nunique()} patients, {len(feats)} features, "
          f"{df['fold'].nunique()} folds, probe_set={args.probe_set}")

    os.makedirs(args.results_dir, exist_ok=True)
    baseline, pred_baseline = run_baseline(df)
    ridge, _, pred_ridge = run_linear(df, feats, "ridge")
    elastic, en_coefs, pred_elastic = run_linear(df, feats, "elastic_net")
    all_preds = [pred_baseline, pred_ridge, pred_elastic]

    ablations = None
    if args.ablations:
        ablations = run_ablations(df, args.probe_set)
        ablations.to_csv(os.path.join(args.results_dir, "ablation_ridge.csv"), index=False)

    all_parts = [baseline, ridge, elastic]
    if not args.skip_cnn:
        idx = load_dataset(args.data_dir, verbose=False)
        path_by_visit = idx.assign(visit_id=idx["path"].map(visit_key)).set_index("visit_id")["path"]
        df = df[df["visit_id"].isin(path_by_visit.index)].reset_index(drop=True)
        df = df.assign(path=df["visit_id"].map(path_by_visit))
        roles = load_roles(args.data_dir)

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        seqs = build_sequences(df, roles, args.embeddings_cache, model)
        cnn, cnn_curves, pred_cnn = run_cnn(df, seqs)
        all_parts.append(cnn)
        all_preds.append(pred_cnn)
        cnn_curves.to_csv(os.path.join(args.results_dir, "cnn_training_curves.csv"), index=False)
        n_stopped = cnn["early_stopped"].sum()
        print(f"CNN early stopping triggered in {n_stopped}/{len(cnn)} folds "
              f"(others ran the full {CNN_MAX_EPOCHS}-epoch cap)")

    results = pd.concat(all_parts, ignore_index=True)
    predictions = pd.concat(all_preds, ignore_index=True)
    predictions.to_csv(os.path.join(args.results_dir, "predictions.csv"), index=False)
    for name, part in (("baseline", baseline), ("ridge", ridge), ("elastic_net", elastic)):
        part.to_csv(os.path.join(args.results_dir, f"{name}_folds.csv"), index=False)
    if not args.skip_cnn:
        cnn.to_csv(os.path.join(args.results_dir, "cnn_folds.csv"), index=False)
    en_coefs.to_csv(os.path.join(args.results_dir, "elastic_net_coefs.csv"), index=False)

    print("\nmean ± std across folds:")
    print(results.groupby("model")[["mae", "rmse", "r2", "spearman_rho"]].agg(["mean", "std"]).round(3))
    write_summary(args.results_dir, results, en_coefs, ablations)

if __name__ == "__main__":
    main()