import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
PROCESSED_DIR = ROOT / "processed"
MODEL_DIR = Path(os.environ.get("HEALTHPAL_MODEL_DIR", ROOT.parent.parent / "models"))
DASEG_SCRIPT = Path(os.environ.get("HEALTHPAL_DASEG_SCRIPT", ROOT.parent.parent / "extract_features" / "daseg_pipeline.py"))
EMBEDDINGS_DIR = Path(os.environ.get("HEALTHPAL_EMBEDDINGS_DIR", ROOT.parent.parent / "embeddings"))
TOPIC_SEED_DIR = ROOT / "topic_cache"   # committed labels for the synthetic demo transcripts (read-only)
TOPIC_SAME_COSINE = 0.65   # topic labels at least this similar count as the same topic
TEXT_COLUMN = "text"
ROLES = ("provider", "patient", "caregiver")   # the speaker roles the app accepts and shows, in display order
MAX_UPLOAD_MB = 20
WORDS_PER_SECOND = 2.5
SHOW_REFERENCE_SCORE = os.environ.get("HEALTHPAL_SHOW_REFERENCE", "") == "1"
DSRS_MAX = 54
BANDS = (("normal_to_mild", 18), ("moderate", 36), ("severe", 54))
