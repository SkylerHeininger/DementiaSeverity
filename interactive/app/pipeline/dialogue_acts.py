import importlib.util
import sys
from pathlib import Path

import pandas as pd

import config
from config import ROLES
from .da_groups import GROUP_ORDER, map_da_to_group

loaded = {}


def load_daseg():
    if not loaded:
        if not config.DASEG_SCRIPT.exists():
            raise FileNotFoundError(f"daseg script not found at {config.DASEG_SCRIPT} (set HEALTHPAL_DASEG_SCRIPT or edit config.py)")
        spec = importlib.util.spec_from_file_location("run_daseg", config.DASEG_SCRIPT)
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except ImportError as error:
            raise ImportError(f"{error} (the app is running under {sys.executable}; start it with the Python from the venv that has the daseg packages)") from error
        loaded["module"] = module
        loaded["pipe"], loaded["tokenizer"] = module.load_pipeline()

    return loaded["module"], loaded["pipe"], loaded["tokenizer"]


def run_daseg(csv_path):
    module, pipe, tokenizer = load_daseg()
    out_dir = config.PROCESSED_DIR / "daseg"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not module.process_file(str(csv_path), str(out_dir), config.TEXT_COLUMN, pipe, tokenizer):
        raise RuntimeError("daseg found nothing usable in this transcript")
    
    return out_dir / f"{csv_path.stem}_with_predictions.csv"


def summarize(words):
    words = words[words["DA_number"] != ""].copy()
    words["turn"] = words["turn"].astype(int)
    segments = []

    for _, seg in words.groupby("DA_number", sort=False):
        label = seg["Proc_DA"].iloc[-1]
        segments.append({
            "turn": int(seg["turn"].iloc[0]), "end_turn": int(seg["turn"].iloc[-1]), "role": seg["role"].iloc[0],
            "label": label, "group": map_da_to_group(label), "text": " ".join(seg[config.TEXT_COLUMN].astype(str)),
        })
    
    segments = [s for s in segments if s["role"] in ROLES]

    turn_groups = {}
    for seg in segments:
        for turn in range(seg["turn"], seg["end_turn"] + 1):
            groups = turn_groups.setdefault(str(turn), [])
            if seg["group"] not in groups:
                groups.append(seg["group"])

    counts = {role: {} for role in ROLES}
    for seg in segments:
        counts[seg["role"]][seg["group"]] = counts[seg["role"]].get(seg["group"], 0) + 1

    n_questions = sum(counts[r].get("questions", 0) for r in ROLES)
    return {
        "segments": segments,
        "turn_groups": turn_groups,
        "groups": [g for g in GROUP_ORDER if any(g in counts[r] for r in ROLES)],
        "role_group_counts": counts,
        "role_totals": {r: sum(counts[r].values()) for r in ROLES},
        "questions_by_role": {r: {"count": counts[r].get("questions", 0),
                                  "share": round(100 * counts[r].get("questions", 0) / n_questions, 1) if n_questions else 0.0}for r in ROLES},
    }


def run(visit, df, results):
    output = run_daseg(Path(visit["csv"]))
    return summarize(pd.read_csv(output, dtype=str, keep_default_na=False))
