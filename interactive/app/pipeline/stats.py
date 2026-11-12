import numpy as np

from config import ROLES

N_BINS = 24


def share(part, whole):
    return round(100 * float(part) / float(whole), 1) if whole else 0.0


def response_gaps(df):
    gaps = {role: [] for role in ROLES}
    for i in range(1, len(df)):
        if df["role"].iloc[i] != df["role"].iloc[i - 1] and df["role"].iloc[i] in gaps:
            gaps[df["role"].iloc[i]].append(max(df["timestamp_seconds"].iloc[i] - df["end_seconds"].iloc[i - 1], 0.0))
    return gaps


def timeline(df, duration):
    edges = np.linspace(0, duration, N_BINS + 1)
    seconds = {role: np.zeros(N_BINS) for role in ROLES}
    for _, row in df.iterrows():
        if row["role"] in seconds:
            seconds[row["role"]] += np.clip(np.minimum(row["end_seconds"], edges[1:]) - np.maximum(row["timestamp_seconds"], edges[:-1]), 0, None)
    return {"edges": [round(float(e), 1) for e in edges], "seconds": {r: [round(float(v), 1) for v in seconds[r]] for r in ROLES}}


def run(visit, df, results):
    df = df.copy()
    df["words"] = df["text"].fillna("").astype(str).str.split().str.len()
    df["seconds"] = (df["end_seconds"] - df["timestamp_seconds"]).clip(lower=0)
    gaps = response_gaps(df)
    rows = []
    for role in ROLES:
        sub = df[df["role"] == role]
        rows.append({
            "role": role,
            "turns": int(len(sub)),
            "words": int(sub["words"].sum()),
            "seconds": round(float(sub["seconds"].sum()), 1),
            "turn_share": share(len(sub), len(df)),
            "word_share": share(sub["words"].sum(), df["words"].sum()),
            "time_share": share(sub["seconds"].sum(), df["seconds"].sum()),
            "mean_words": round(float(sub["words"].mean()), 1) if len(sub) else 0.0,
            "median_words": float(sub["words"].median()) if len(sub) else 0.0,
            "longest_turn_words": int(sub["words"].max()) if len(sub) else 0,
            "median_response_seconds": round(float(np.median(gaps[role])), 1) if gaps[role] else None,
        })
    
    duration = float(df["end_seconds"].max())
    return {"duration_seconds": round(duration, 1), "n_turns": int(len(df)), "n_words": int(df["words"].sum()),
            "roles": rows, "timeline": timeline(df, duration)}