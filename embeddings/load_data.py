"""
Forrest Leung
10/26/26

Shared loading from external data folder.

To run:
python embeddings/load_data.py "$DATA"
"""
import argparse
import csv
import glob
import json
import os
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

TRANSCRIPT_PATTERNS = [
    ("NYU", "VCs1_NYU_Transcriptions/V[0-9]* English/*.csv"),
    ("Dartmouth", "VCs1_Transcriptions/Visit [0-9]*/*.csv"),
    ("Dartmouth", "VCs1_visit1/*.csv"),   # listed last so the Visit 1 copy wins on duplicates
]
SEVERITY_GLOB = "VCs1_severity/processed/*.xlsx"
ROLES_CSV = "VCs1_Transcriptions/speaker_roles.csv"

# Record ids listed here (one per line, "#" comments allowed) are kept out of the pool.
BLACKLIST_FILE = "blacklist.txt"

# Mapping of severity timepoints to the target months after the patient's first visit.
TIMEPOINT_MONTHS = {0: 0, 2: 6, 3: 12}
WINDOW_MONTHS = 3
STUDY_START = pd.Timestamp("2022-01-01")

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

# Utility functions for reading and normalizing data files.
def read_table(path):
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, csv.Error):
            return read_table_lenient(path, enc)

# Clean column names to a consistent format.
def clean_col(c):
    c = str(c).replace("﻿", "").strip().strip('"').strip("'").strip().lower()
    return re.sub(r"[\s\-]+", "_", c)

# Lenient CSV reading for files with badly quoted text.
def read_table_lenient(path, encoding):
    import csv
    with open(path, encoding=encoding, errors="replace", newline="") as f:
        lines = f.read().splitlines()
    # Header may not be on the first line
    start = next((i for i, l in enumerate(lines[:10]) if "turn_index" in clean_col(l)), 0)
    lines = lines[start:]
    if not lines:
        return pd.DataFrame()
    delim = max(["\t", ",", ";"], key=lines[0].count)
    rows = list(csv.reader(lines, delimiter=delim, strict=False))
    header, body = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]
    n = len(header)
    fixed = [r[:n - 1] + [delim.join(r[n - 1:])] if len(r) > n else r + [""] * (n - len(r))
             for r in body]
    return pd.DataFrame(fixed, columns=header)

# Normalize transcript file names for consistent matching with severity data.
def key(name):
    stem = Path(str(name).strip()).name
    stem = re.sub(r"\.(csv|docx|txt)$", "", stem.strip(), flags=re.I)
    return re.sub(r"\s+", " ", stem).strip().lower()

def normalize_role(r):
    r = str(r).strip().lower()
    if r.startswith(("pat", "pt")):
        return "patient"
    if r.startswith(("care", "cg", "family", "spouse", "companion")):
        return "caregiver"
    if r.startswith(("prov", "doc", "dr", "clin", "nurse", "phys")):
        return "provider"
    if r.startswith("coord"):
        return "coordinator"
    return "other"

# Normalize patient IDs for consistent matching with severity data.
def norm_id(x):
    if pd.isna(x):
        return None
    m = re.search(r"([A-Za-z]\d{2})[-_ ]?(\d{3,4})", str(x))
    if m:
        return f"{m.group(1).upper()}-{int(m.group(2))}"
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())

# Record ids excluded from the pool entirely, read from <data_dir>/blacklist.txt by default.
# Ids are normalized the same way as everywhere else, so D01_0001, D01-1 and D01 0001 all match.
def load_blacklist(data_dir, path=None, verbose=False):
    path = path or os.path.join(data_dir, BLACKLIST_FILE)
    if not os.path.exists(path):
        if verbose:
            print(f"no blacklist at {path}; nothing excluded by id")
        return frozenset()
    ids = {norm_id(line.split("#")[0].strip()) for line in Path(path).read_text().splitlines()}
    ids.discard(None)
    ids.discard("")
    if verbose:
        print(f"blacklist: {len(ids)} record id(s) from {path}")
    return frozenset(ids)

# Drop rows whose (already normalized) id column is on the blacklist.
def drop_blacklisted(df, blacklist, id_col="pid_norm", what="row", verbose=False):
    if not blacklist or df.empty:
        return df
    hit = df[id_col].isin(blacklist)
    if verbose and hit.any():
        print(f"  blacklist: dropped {int(hit.sum())} {what}(s) for "
              f"{df.loc[hit, id_col].nunique()} record id(s)")
    return df[~hit].reset_index(drop=True)

# Parse dates from text strings in various formats.
def parse_date(text):
    m = re.search(r"(\d{1,2})\s*([A-Za-z]{3,9})\s*(\d{4}|\d{2})(?!\d)", text)
    if not m:
        return None
    day, mon, yr = int(m.group(1)), MONTHS.get(m.group(2)[:3].lower()), int(m.group(3))
    if mon is None:
        return None
    yr = yr + 2000 if yr < 100 else yr
    try:
        return date(yr, mon, day)
    except ValueError:
        return None

# Parse patient ID, visit number, and visit date from transcript file names.
def parse_file_name(path):
    name = Path(path).name
    pid = re.match(r"\s*([A-Za-z]\d{2}[-_ ]?\d{3,4})", name)
    visit = re.search(r"\bV(\d+)\b", name)
    after_visit = name[visit.end():] if visit else name
    return (norm_id(pid.group(1)) if pid else None,
            int(visit.group(1)) if visit else None,
            parse_date(after_visit))

def find_transcripts(data_dir, blacklist=None, verbose=False):
    blacklist = load_blacklist(data_dir) if blacklist is None else blacklist
    rows = []
    for site, pattern in TRANSCRIPT_PATTERNS:
        for p in sorted(glob.glob(os.path.join(data_dir, pattern))):
            if p.endswith("_llm_labels.csv") or "bad_diarization" in p:
                continue
            pid, visit, d = parse_file_name(p)
            rows.append({"path": p, "site": site, "key": key(p),
                         "pid_norm": pid, "visit": visit, "date": d})
    tr = pd.DataFrame(rows, columns=["path", "site", "key", "pid_norm", "visit", "date"])
    tr["dup_name"] = tr["key"].duplicated(keep="first")
    tr = tr[~tr["dup_name"]].drop(columns="dup_name").reset_index(drop=True)
    tr = drop_blacklisted(tr, blacklist, what="transcript file", verbose=verbose)

    every = []
    for root in ("VCs1_NYU_Transcriptions", "VCs1_Transcriptions", "VCs1_visit1"):
        for p in glob.glob(os.path.join(data_dir, root, "**", "*.*"), recursive=True):
            if p.lower().endswith((".csv", ".docx")) and not p.endswith("_llm_labels.csv"):
                pid, visit, d = parse_file_name(p)
                if pid and d:
                    every.append({"pid_norm": pid, "visit": visit, "date": d})
    every = pd.DataFrame(every, columns=["pid_norm", "visit", "date"])
    every["date"] = pd.to_datetime(every["date"])
    tr["date"] = pd.to_datetime(tr["date"])

    # Filter out dates that are before the study start date.
    every = every[every["date"] >= STUDY_START]
    tr.loc[tr["date"] < STUDY_START, "date"] = pd.NaT
    v1 = every[every["visit"] == 1].groupby("pid_norm")["date"].min()
    first = every.groupby("pid_norm")["date"].min()
    base = tr["pid_norm"].map(v1).fillna(tr["pid_norm"].map(first))
    tr["months"] = (tr["date"] - base).dt.days / 30.44
    tr.loc[tr["months"] < -1, "months"] = None
    return tr

def load_severity(data_dir, blacklist=None, verbose=False):
    blacklist = load_blacklist(data_dir) if blacklist is None else blacklist
    frames = []
    for p in sorted(glob.glob(os.path.join(data_dir, SEVERITY_GLOB))):
        if Path(p).name.startswith("~$"):
            continue
        f = pd.read_excel(p)
        f["source_file"] = Path(p).name
        frames.append(f)
    if not frames:
        raise SystemExit(f"no severity files found at {os.path.join(data_dir, SEVERITY_GLOB)}")
    df = pd.concat(frames, ignore_index=True)
    df["pid_norm"] = df["record_id_clean"].map(norm_id)
    df = drop_blacklisted(df, blacklist, what="severity row", verbose=verbose)
    df["timepoint"] = pd.to_numeric(
        df["Event Name"].astype(str).str.extract(r"\bT(\d+)\b")[0], errors="coerce")
    # matched_transcript can list several files: "V2 English/a.csv;V2 English/a.docx;..."
    df["cand_keys"] = df["matched_transcript"].map(
        lambda x: list(dict.fromkeys(key(p) for p in str(x).split(";") if p.strip()))
        if pd.notna(x) and str(x).strip() else [])
    df["key"] = df["cand_keys"].map(lambda ks: ks[0] if ks else None)
    return df

def load_roles(data_dir):
    p = os.path.join(data_dir, ROLES_CSV)
    if not os.path.exists(p):
        return None
    r = read_table(p)
    r["key"] = r["file"].map(key)
    r["speaker"] = r["speaker"].astype(str).str.strip()
    r["ground_truth_role"] = r["ground_truth_role"].map(normalize_role)
    return r[["key", "speaker", "ground_truth_role"]].drop_duplicates(["key", "speaker"])

# Load a single transcript file and optionally apply ground-truth roles from the roles DataFrame.
def load_transcript(path, roles=None):
    t = read_table(path)
    t.columns = [clean_col(c) for c in t.columns]
    if "turn_index" not in t.columns:
        t = read_table_lenient(path, "utf-8-sig")
        t.columns = [clean_col(c) for c in t.columns]
    missing = {"turn_index", "speaker", "role", "text"} - set(t.columns)
    if missing:
        raise KeyError(f"missing columns {sorted(missing)}; found {len(t.columns)} columns "
                       f"(e.g. {sorted(t.columns)[:6]})")
    t["speaker"] = t["speaker"].astype(str).str.strip()
    t["role"] = t["role"].map(normalize_role)
    t["role_source"] = "transcript"
    t["role_disagrees"] = False
    if roles is not None:
        gt = roles[roles["key"] == key(path)].set_index("speaker")["ground_truth_role"]
        if len(gt):
            mapped = t["speaker"].map(gt)
            has = mapped.notna()
            t.loc[has, "role_disagrees"] = t.loc[has, "role"] != mapped[has]
            t.loc[has, "role"] = mapped[has]
            t.loc[has, "role_source"] = "ground_truth"
    t["text"] = t["text"].fillna("").astype(str)
    if "word_count" not in t:
        t["word_count"] = t["text"].str.split().str.len()
    t["word_count"] = pd.to_numeric(t["word_count"], errors="coerce").fillna(0)
    t["turn_index"] = pd.to_numeric(t["turn_index"], errors="coerce")
    t["timestamp_seconds"] = pd.to_numeric(t["timestamp_seconds"], errors="coerce")
    return t.sort_values("turn_index").reset_index(drop=True)

# Choose the transcript that best fits a given severity timepoint.
def choose_for_timepoint(cands, tp):
    if cands.empty:
        return None
    if pd.isna(tp) or int(tp) == 0:
        firsts = cands[cands["visit"] == 1]
        if len(firsts):
            return firsts.index[0]
        # first visit not usable (e.g. bilingual): the earliest visit within the window
        pool = cands[cands["months"] <= WINDOW_MONTHS]
        return pool.sort_values("months").index[0] if len(pool) else None
    target = TIMEPOINT_MONTHS.get(int(tp))
    if target is None:
        return None
    dist = (cands["months"] - target).abs().dropna()
    dist = dist[dist <= WINDOW_MONTHS]
    return dist.idxmin() if len(dist) else None

# Build an index of severity records paired with the most appropriate transcripts.
def build_index(data_dir, verbose=True, report_dir=None, blacklist=None):
    blacklist = load_blacklist(data_dir, verbose=verbose) if blacklist is None else blacklist
    tr = find_transcripts(data_dir, blacklist, verbose=verbose)
    sev = load_severity(data_dir, blacklist, verbose=verbose)
    scored = sev.dropna(subset=["total_score"]).copy().reset_index(drop=True)

    tr_by_key = {k: i for i, k in enumerate(tr["key"])}
    listed = scored["cand_keys"].map(lambda ks: [tr_by_key[k] for k in ks if k in tr_by_key])
    n_listed = listed.str.len()

    def fits_date(j, tp):
        """A follow-up survey (T2/T3) must pair with a transcript near its target month."""
        if pd.isna(tp) or int(tp) == 0:
            return True
        target, m = TIMEPOINT_MONTHS.get(int(tp)), tr.at[j, "months"]
        return target is not None and pd.notna(m) and abs(m - target) <= WINDOW_MONTHS

    used, pairs, retry = set(), {}, []
    n_rejected = 0
    methods = ["matched_transcript", "matched_transcript, chosen by date",
               "patient ID + date (fallback)"]
    # Pass 1: exactly one listed CSV
    for i in scored.index[n_listed == 1]:
        j = listed[i][0]
        if j in used:
            retry.append(i)
        elif not fits_date(j, scored.at[i, "timepoint"]):
            n_rejected += 1          # e.g. a 6-month survey paired with the baseline transcript
            retry.append(i)
        else:
            used.add(j)
            pairs[i] = (j, methods[0])
    # Pass 2: several listed CSVs -> the one whose date fits
    for i in scored.index[n_listed > 1]:
        cand = [j for j in listed[i] if j not in used]
        pick = choose_for_timepoint(tr.loc[cand], scored.at[i, "timepoint"])
        if pick is None:
            retry.append(i)
        else:
            used.add(pick)
            pairs[i] = (pick, methods[1])
    # Pass 3: nothing usable listed -> any of the patient's transcripts whose date fits
    for i in list(scored.index[n_listed == 0]) + retry:
        pool = tr[(tr["pid_norm"] == scored.at[i, "pid_norm"]) & ~tr.index.isin(used)]
        pick = choose_for_timepoint(pool, scored.at[i, "timepoint"])
        if pick is not None:
            used.add(pick)
            pairs[i] = (pick, methods[2])
    passes = [(m, None) for m in methods]

    found = scored.loc[list(pairs)].copy()
    found["tr_idx"] = [pairs[i][0] for i in found.index]
    found["match_method"] = [pairs[i][1] for i in found.index]
    found = found.drop(columns=["key"]).merge(
        tr[["path", "site", "key", "visit", "months"]], left_on="tr_idx", right_index=True)

    # Flag visits that are out of the acceptable window relative to their target timepoint.
    target_months = found["timepoint"].map(TIMEPOINT_MONTHS)
    found["months_gap_from_target"] = (found["months"] - target_months).abs()
    found["timepoint_out_of_window"] = (found["months_gap_from_target"] > WINDOW_MONTHS).fillna(False)

    unmatched_sev = scored[~scored.index.isin(pairs)]
    unmatched_tr = tr[~tr.index.isin(used)]

    if verbose:
        print(f"transcript CSVs on disk (English visits, excluding bad_diarization): {len(tr)}"
              f"  [{', '.join(f'{s}: {n}' for s, n in tr['site'].value_counts().items())}]")
        print(f"  dates parsed from {tr['date'].notna().sum()}/{len(tr)} file names; "
              f"{tr['months'].isna().sum()} transcripts have no usable months-since-baseline")
        print(f"severity rows: {len(sev)} | with total_score: {len(scored)}")
        for method, _ in passes:
            print(f"  matched by {method}: {(found['match_method'] == method).sum()}")
        if n_rejected:
            print(f"  rejected {n_rejected} team pairings of a T2/T3 survey with a transcript "
                  f"outside ±{WINDOW_MONTHS} months of its target; searched by date instead")
        n_oow = int(found["timepoint_out_of_window"].sum())
        if n_oow:
            by_tp = found.loc[found["timepoint_out_of_window"], "timepoint"].value_counts().sort_index()
            print(f"  flagged (not dropped) {n_oow} visits more than ±{WINDOW_MONTHS} months from "
                  f"their nominal target, incl. T0 which has no window check during matching: "
                  f"{dict((f'T{int(k)}', int(v)) for k, v in by_tp.items())}")
        per_patient = found.groupby("record_id_clean")["timepoint"].nunique()
        print(f"patients with 1 / 2 / 3 timepoints: {(per_patient == 1).sum()} / "
              f"{(per_patient == 2).sum()} / {(per_patient == 3).sum()}")
        print(f"unmatched: {len(unmatched_sev)} scored severity rows, {len(unmatched_tr)} transcript CSVs")
        print(f"usable: {len(found)} visits from {found['record_id_clean'].nunique()} patients  "
              f"[{', '.join(f'{s}: {n}' for s, n in found['site'].value_counts().items())}]")
        print("usable visits by timepoint:",
              {int(k): int(v) for k, v in found["timepoint"].value_counts().sort_index().items()})

    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        unmatched_sev[["source_file", "record_id_clean", "Event Name", "matched_transcript"]].to_csv(
            os.path.join(report_dir, "unmatched_severity_rows.csv"), index=False)
        unmatched_tr.assign(path=unmatched_tr["path"].map(lambda p: os.path.relpath(p, data_dir)))[
            ["path", "site", "pid_norm", "visit", "date", "months"]].to_csv(
            os.path.join(report_dir, "unmatched_transcripts.csv"), index=False)
        if verbose:
            print(f"unmatched lists written to {report_dir} (they contain IDs; keep them local)")
    return found.drop(columns=["tr_idx"]).reset_index(drop=True)

# Baseline visits: one visit per patient, the earliest timepoint available.
def baseline_visits(idx):
    return (idx.sort_values(["record_id_clean", "timepoint"])
               .drop_duplicates("record_id_clean")
               .reset_index(drop=True))


# Shared turn-embedding cache for efficiency.
EMBED_MODEL = "FremyCompany/BioLORD-2023"
FEATURE_ROLES = ["patient", "caregiver", "provider", "coordinator"]
OTHER_ROLE = "other"
ROLE_CHANNELS = FEATURE_ROLES + [OTHER_ROLE]

def embedding_cache_file(cache_dir, visit_id):
    return os.path.join(cache_dir, f"{visit_id}.npz")

# Get turn embeddings for a given transcript, using a cache to avoid recomputation.
def get_turn_embeddings(path, roles, cache_dir, model):
    vid = key(path)
    f = embedding_cache_file(cache_dir, vid)
    if os.path.exists(f):
        d = np.load(f)
        return d["emb"], d["role"]
    t = load_transcript(path, roles)
    emb = model.encode(t["text"].tolist(), normalize_embeddings=True, batch_size=64,
                       show_progress_bar=False).astype(np.float32)
    role = np.zeros((len(t), len(ROLE_CHANNELS)), dtype=np.float32)
    for j, r in enumerate(ROLE_CHANNELS):
        role[:, j] = (t["role"] == r).to_numpy()
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(f, emb=emb, role=role)
    return emb, role


# Language detection for individual turns in a transcript.
LANG_EN = set("""the a an and or but of to in on at for with is are was were be been being am
i you he she it we they me him her us them my your his its our their this that these those
what when where why how who which do does did don't didn't not no yes yeah okay ok so just
like have has had can could would should will i'm it's that's there here about if then
because very really know think go going get got from up out all some""".split())
LANG_ES = set("""el la los las un una unos unas y o pero de del a al en con por para que qué es
son está están estaba fue ser estar soy eres somos yo tú usted él ella nosotros ellos ellas
me te se le les lo mi mis su sus nuestro este esta esto ese esa eso aquí allí cuando donde
dónde porque cómo como muy sí no bueno pues entonces también todo nada algo hay tiene tengo
puede quiero sabe sé ahora ya más mucho poco gracias señora señor mamá doctor doctora""".split())
NON_LATIN_RE = re.compile(r"[Ѐ-ӿ֐-ۿऀ-ॿ぀-ヿ㐀-鿿가-힯]")
LANG_WORD_RE = re.compile(r"[a-záéíóúñü']+")
MIN_WORDS_FOR_LANG = 4
MIN_LANG_CONFIDENCE = 0.15

# Determine the language of a single turn in a transcript.
def turn_language(text):
    if NON_LATIN_RE.search(text):
        return "other", 1.0
    words = LANG_WORD_RE.findall(text.lower())
    if len(words) < MIN_WORDS_FOR_LANG:
        return None, 0.0
    en = sum(w in LANG_EN for w in words)
    es = sum(w in LANG_ES and w not in LANG_EN for w in words) + 2 * len(re.findall(r"[ñ¿¡]", text))
    total = len(words)
    if es > en:
        label, conf = "spanish", (es - en) / total
    elif en / total < 0.08 and es == 0:
        label, conf = "other", 1 - en / total       # neither language's function words
    else:
        label, conf = "english", en / total
    conf = min(conf, 1.0)
    return (label, conf) if conf >= MIN_LANG_CONFIDENCE else (None, conf)

# Compute language statistics for a given transcript.
def visit_language_stats(t):
    labels = t["text"].map(lambda x: turn_language(x)[0])
    judged = labels.notna()
    non_en = labels.isin(["spanish", "other"])
    words = t["word_count"]
    share = words[non_en].sum() / max(words[judged].sum(), 1)
    return float(share), float(judged.mean()) if len(t) else (0.0, 0.0)

# Annotate visits with caregiver presence and non-English language statistics.
def annotate_visits(idx, roles=None, require_caregiver=True, language_threshold=0.10, verbose=True):
    has_cg, pct_non_en, unreadable = [], [], []
    for i, row in idx.iterrows():
        try:
            t = load_transcript(row["path"], roles)
        except Exception:
            has_cg.append(False)
            pct_non_en.append(np.nan)
            unreadable.append(i)
            continue
        has_cg.append(bool((t["role"] == "caregiver").any()))
        share, _ = visit_language_stats(t)
        pct_non_en.append(share)

    idx = idx.copy()
    idx["has_caregiver"] = has_cg
    idx["pct_non_english"] = pct_non_en
    idx["flag_non_english"] = idx["pct_non_english"] >= language_threshold
    n_total = len(idx)

    drop_caregiver = (~idx["has_caregiver"]) if require_caregiver else pd.Series(False, index=idx.index)
    kept = idx[~drop_caregiver]
    drop_language = kept["flag_non_english"] | kept.index.isin(unreadable)
    final = kept[~drop_language].reset_index(drop=True)

    if verbose:
        if unreadable:
            print(f"  {len(unreadable)} matched transcripts could not be read; excluded")
        print(f"caregiver filter: dropped {int(drop_caregiver.sum())}/{n_total} visits "
              f"with no caregiver turns (require_caregiver={require_caregiver})")
        print(f"language filter: dropped {int(drop_language.sum())}/{len(kept)} visits "
              f">= {language_threshold:.0%} non-English words (or unreadable)")
        print(f"usable after filters: {len(final)} visits from "
              f"{final['record_id_clean'].nunique()} patients")
    return final


# Assign patient-grouped, severity-stratified folds for cross-validation.
FOLD_FILE = "derived/fold_assignment.json"

# Assigns folds for cross-validation, ensuring that all visits from the same patient
def assign_folds(idx, data_dir, k=5, seed=0, band_col="severity_category"):
    path = os.path.join(data_dir, FOLD_FILE)
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        if cached.get("k") == k and cached.get("seed") == seed:
            return idx["record_id_clean"].map(cached["fold_of"]).astype("Int64")

    fold_of = compute_folds(idx, k=k, seed=seed, band_col=band_col)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"k": k, "seed": seed, "fold_of": fold_of}, f, indent=2)
    return idx["record_id_clean"].map(fold_of).astype("Int64")

# Same split as assign_folds, computed in memory with no fold-file read or write: {patient: fold}.
# For repeated CV over many seeds, where caching each seed would overwrite the canonical fold file.
def compute_folds(idx, k=5, seed=0, band_col="severity_category"):
    # Ensure that each patient appears only once in the list of patients for stratification.
    patients = idx.sort_values(["record_id_clean", "timepoint"]).drop_duplicates("record_id_clean")
    groups = patients["record_id_clean"].to_numpy()
    y = patients[band_col].to_numpy() if band_col in patients else None
    X = np.zeros(len(patients))
    try:
        if y is None or pd.isna(y).any():
            raise ValueError("no usable severity band for stratification")
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y, groups))
    except ValueError:
        splitter = GroupKFold(n_splits=k)
        splits = list(splitter.split(X, groups=groups))

    fold_of = {}
    for fi, (_, test_idx) in enumerate(splits):
        for pid in groups[test_idx]:
            fold_of[str(pid)] = fi
    return fold_of

# Load the canonical visit-level dataset with all annotations and folds applied.
def load_dataset(data_dir, require_caregiver=True, language_threshold=0.10,
                  k=5, seed=0, report_dir=None, verbose=True, blacklist=None):
    report_dir = report_dir or os.path.join(data_dir, "derived")
    idx = build_index(data_dir, verbose=verbose, report_dir=report_dir, blacklist=blacklist)
    roles = load_roles(data_dir)
    idx = annotate_visits(idx, roles, require_caregiver, language_threshold, verbose)
    idx["fold"] = assign_folds(idx, data_dir, k=k, seed=seed)
    return idx

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--no-require-caregiver", dest="require_caregiver", action="store_false",
                    help="keep visits with no caregiver turns too (default: drop them)")
    ap.add_argument("--language-threshold", type=float, default=0.10,
                    help="drop a visit if at least this share of judged words are non-English")
    ap.add_argument("--blacklist", default=None,
                    help=f"file of record ids to exclude, one per line "
                         f"(default: {BLACKLIST_FILE} in data_dir, if present)")
    ap.add_argument("--no-blacklist", dest="use_blacklist", action="store_false",
                    help="ignore the blacklist file and keep every record id")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="also write the visit-level table (no transcript paths) to this CSV")
    args = ap.parse_args()
    blacklist = (load_blacklist(args.data_dir, args.blacklist, verbose=True)
                 if args.use_blacklist else frozenset())

    df = load_dataset(args.data_dir, require_caregiver=args.require_caregiver,
                      language_threshold=args.language_threshold, k=args.k, seed=args.seed,
                      blacklist=blacklist)
    print("\nfold sizes (visits):", {int(k): int(v) for k, v in df["fold"].value_counts().sort_index().items()})
    print("fold sizes (patients):", {int(k): int(v) for k, v in
          df.groupby("fold")["record_id_clean"].nunique().sort_index().items()})
    if args.out:
        df.drop(columns=["path"]).to_csv(args.out, index=False)
        print(f"wrote {args.out}")

if __name__ == "__main__":
    main()