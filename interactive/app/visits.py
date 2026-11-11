import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config

ROLES = ("provider", "patient", "caregiver")
ROLE_ALIASES = {"clinician": "provider", "doctor": "provider", "physician": "provider"}
REQUIRED_COLUMNS = ("role", "text", "timestamp_seconds")
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def meta_path(csv_path):
    return csv_path.with_name(f"{csv_path.stem}_meta.json")


def make_visit(csv_path):
    path = meta_path(csv_path)
    meta = json.loads(path.read_text()) if path.exists() else {}
    name = meta.get("filename", csv_path.name)
    return {
        "id": csv_path.stem,
        "csv": str(csv_path),
        "meta": meta,
        "title": name,
        "uploaded": meta.get("uploaded_at", "").replace("T", " "),
        "n_turns": meta.get("n_turns"),
        "duration_minutes": round(meta["duration_seconds"] / 60, 1) if meta.get("duration_seconds") else None,
        "end_estimated": meta.get("end_estimated", False),
        "media": [{"filename": name, "content_type": "text/csv", "size_bytes": csv_path.stat().st_size}],
    }


def get_visit(visit_id):
    if not ID_PATTERN.match(visit_id):
        return None

    csv_path = config.UPLOAD_DIR / f"{visit_id}.csv"
    return make_visit(csv_path) if csv_path.exists() else None


def new_id(filename):
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(filename).stem).strip("-")[:40] or "transcript"
    return f"{stem}-{uuid.uuid4().hex[:8]}"


def normalize(raw):
    raw = raw.rename(columns=lambda c: str(c).strip().lower())
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. The file has: {', '.join(raw.columns)}.")

    df = raw.copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("The transcript has no text.")

    df["role"] = df["role"].fillna("").astype(str).str.strip().str.lower().replace(ROLE_ALIASES)
    unknown = sorted(set(df["role"]) - set(ROLES))
    if unknown:
        raise ValueError(f"Unrecognized role value(s): {', '.join(repr(r) for r in unknown)}. Every role must be provider, patient, or caregiver.")

    start = pd.to_numeric(df["timestamp_seconds"], errors="coerce")
    if start.isna().any():
        raise ValueError("timestamp_seconds must be a number on every turn.")

    given = pd.to_numeric(df["end_seconds"], errors="coerce") if "end_seconds" in df.columns else pd.Series(np.nan, index=df.index)
    guess = np.fmin(start + df["text"].str.split().str.len() / config.WORDS_PER_SECOND, start.shift(-1))
    end = np.maximum(given.fillna(guess), start)

    speaker = df["speaker"].fillna("").astype(str).str.strip() if "speaker" in df.columns else pd.Series("", index=df.index)
    speaker = speaker.where(speaker != "", df["role"].str.title())

    out = pd.DataFrame({
        "turn": np.arange(len(df)), "role": df["role"], "speaker": speaker, "text": df["text"],
        "timestamp_seconds": start.round(3), "end_seconds": end.round(3),
    })
    return out, bool(given.isna().any())


def save_upload(file):
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise ValueError("Upload a CSV file (.csv).")

    try:
        raw = pd.read_csv(file.stream, dtype=str, keep_default_na=False, encoding="utf-8-sig", encoding_errors="replace")
    except Exception as error:
        raise ValueError(f"Could not read the file as a CSV: {error}") from error

    df, end_estimated = normalize(raw)
    visit_id = new_id(filename)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.UPLOAD_DIR / f"{visit_id}.csv"
    df.to_csv(csv_path, index=False)
    meta = {
        "filename": Path(filename).name, "uploaded_at": datetime.now().isoformat(timespec="minutes"),
        "n_turns": int(len(df)), "duration_seconds": float(df["end_seconds"].max()), "end_estimated": end_estimated,
    }
    meta_path(csv_path).write_text(json.dumps(meta))
    return make_visit(csv_path)


def load_frame(visit):
    return pd.read_csv(visit["csv"])


def turns_payload(df):
    return [
        {"i": i, "role": row["role"], "speaker": row.get("speaker", row["role"]), "text": str(row["text"]),
         "start": float(row["timestamp_seconds"]), "end": float(row["end_seconds"])}
        for i, (_, row) in enumerate(df.iterrows())
    ]
