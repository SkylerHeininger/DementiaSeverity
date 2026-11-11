import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
PROCESSED_DIR = ROOT / "processed"
MODEL_DIR = Path(os.environ.get("HEALTHPAL_MODEL_DIR", ROOT.parent.parent / "models"))
DASEG_SCRIPT = Path(os.environ.get("HEALTHPAL_DASEG_SCRIPT", ROOT.parent.parent / "extract_features" / "daseg_pipeline.py"))
TEXT_COLUMN = "text"
MAX_UPLOAD_MB = 20
WORDS_PER_SECOND = 2.5
SHOW_REFERENCE_SCORE = os.environ.get("HEALTHPAL_SHOW_REFERENCE", "") == "1"
DSRS_MAX = 54
BANDS = (("normal_to_mild", 18), ("moderate", 36), ("severe", 54))
