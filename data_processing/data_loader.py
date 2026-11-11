"""
Skyler Heininger

For loading dataset from all the transcripts, DSRS files, etc

This requires all transcripts (all visits) to be in one folder, to simplify across the 
different folder structures from each site. The visit (V1, V2, ...) and the visit date (DDMonYY)
are parsed from the filename. T0 DSRS is matched to the V1 transcript, T2 to the later visit
dated around 6 months after V1, and T3 to the later visit dated around 12 months after V1.

Claude Sonnet 5.5 was used to help troubleshoot some issues loading different forms of data.
This was largely because we got access to more data, in different forms, thruoghout the project.
So, Sonnet was used to help modify the existing pipeline after it broke with new data
"""

import argparse
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_ROLES = ("provider", "patient", "caregiver", "coordinator")
SITE_BY_PREFIX = {"D": "dartmouth", "N": "nyu"}
REQUIRED_SEVERITY_COLS = ("total_score", "record_id_clean", "severity_category")

# We match the timepoints to nearest visit at 6 months and a year
TIMEPOINT_MONTHS = {"T2": 6, "T3": 12}
DEFAULT_TOLERANCES = {"T2": 2.0, "T3": 4.0}

# Regexes
_RECORD_ID_RE = re.compile(r"^(?P<record_id>[^\s.]+)")
_VISIT_RE = re.compile(r"(?<![A-Za-z0-9])V(\d+)(?![A-Za-z0-9])", re.IGNORECASE)
_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})\s?([A-Za-z]{3,9})(\d{4}|\d{2})(?!\d)")
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_SEVERITY_RE = re.compile(r"^(?P<site>dartmouth|nyu)\s*DSRS_?T(?P<tp>\d+)", re.IGNORECASE)


def normalize_id(value):
    text = str(value if value is not None else "").strip().upper()
    if text in ("", "NAN", "NONE"):
        return ""
    tokens = [t for t in re.split(r"[^A-Z0-9]+", text) if t]
    if not tokens:
        return ""
    if len(tokens) >= 2 and tokens[1].isdigit():
        num = int(tokens[1])
        return f"{tokens[0]}_{num:04d}" if tokens[0][0] == "D" else f"{tokens[0]}-{num:03d}"
    return "-".join(tokens)


def parse_date(name):
    m = _DATE_RE.search(name)
    if not m:
        return pd.NaT
    day, mon, yr = m.groups()
    month = MONTHS.get(mon[:3].lower())
    if month is None:
        return pd.NaT
    try:
        return pd.Timestamp(year=int(yr) + (2000 if len(yr) == 2 else 0), month=month, day=int(day))
    except ValueError:
        return pd.NaT


def discover_transcripts(transcripts_dir, verbose=True):
    files = sorted(p for p in transcripts_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    df = pd.DataFrame({"path": files})
    df["name"] = df["path"].map(lambda p: p.name)
    df["record_id"] = df["name"].str.extract(_RECORD_ID_RE)["record_id"].map(normalize_id)
    df["site"] = df["record_id"].str[:1].map(SITE_BY_PREFIX).fillna("unknown")
    df["is_bilingual"] = df["name"].str.lower().str.contains("bilingual")
    df["visit"] = pd.to_numeric(df["name"].str.extract(_VISIT_RE)[0], errors="coerce")
    df["date"] = df["name"].map(parse_date)

    if verbose:
        print(f"Found {len(df)} transcript file(s) in {transcripts_dir}")
        for _, row in df.sort_values(["record_id", "visit", "name"]).iterrows():
            visit = f"V{int(row['visit'])}" if pd.notna(row["visit"]) else "V?"
            date = row["date"].date().isoformat() if pd.notna(row["date"]) else "no date"
            print(f"  {row['name']}  ->  id {row['record_id']}  {row['site']}  {visit}  {date}")
    unknown = df["site"] == "unknown"
    no_visit = ~unknown & df["visit"].isna()
    if verbose:
        for name in df.loc[unknown, "name"]:
            print(f"  skipped, site unknown (doesn't start with D or N): {name}")
        for name in df.loc[no_visit, "name"]:
            print(f"  skipped, no visit (V#) in filename: {name}")
    return df.loc[~unknown & ~no_visit].reset_index(drop=True)


def group_visits(transcripts, verbose=True):
    people = {}
    for (site, rid, visit), grp in transcripts.groupby(["site", "record_id", "visit"]):
        if grp["is_bilingual"].any() and not grp["is_bilingual"].all():
            if verbose:
                print(f"  {rid} V{int(visit)}: English and Bilingual versions found, using English only")
            grp = grp[~grp["is_bilingual"]]
        dates = grp["date"].dropna()
        if verbose and len(dates) and (dates.max() - dates.min()).days > 31:
            print(f"  {rid} V{int(visit)}: filenames disagree on date ({'; '.join(grp['name'])}), using earliest")
        people.setdefault((site, rid), {})[int(visit)] = {
            "date": dates.min() if len(dates) else pd.NaT,
            "files": grp,
        }
    return people


def find_severity_files(severity_dir):
    files = {}
    for path in sorted(severity_dir.glob("*.xlsx")) + sorted(severity_dir.glob("*.csv")):
        if path.name.startswith("~$"):
            continue
        m = _SEVERITY_RE.match(path.stem.strip())
        if not m:
            continue
        key = (m.group("site").lower(), f"T{int(m.group('tp'))}")
        if key in files:
            print(f"  duplicate severity file for {key[0]} {key[1]}, kept {files[key].name}, ignored {path.name}")
            continue
        files[key] = path
    return files


def safe_column_name(name):
    return "sev_" + re.sub(r"\W+", "_", str(name)).strip("_")


def load_severity(path, site, extra_columns=()):
    if not path.is_file():
        raise SystemExit(f"{site} severity file not found: {path}")
    sev = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)

    missing = [c for c in REQUIRED_SEVERITY_COLS if c not in sev.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing column(s) {missing}. This needs the PROCESSED output of "
            f"process_dsrs.py (the file in its 'processed/' subdirectory), not the raw DSRS "
            f"export. Columns found: {list(sev.columns)}"
        )

    out = pd.DataFrame({
        "site": site,
        "record_id": sev["record_id_clean"].map(normalize_id),
        "total_score": pd.to_numeric(sev["total_score"], errors="coerce"),
        "severity_category": sev["severity_category"],
    })
    for col in extra_columns:
        if col in sev.columns:
            out[safe_column_name(col)] = sev[col].to_numpy()
        else:
            print(f"{path.name}: extra column '{col}' not found, left empty. Columns found: {list(sev.columns)}")
            out[safe_column_name(col)] = ""

    no_score = out["total_score"].isna()
    for rid in out.loc[no_score, "record_id"]:
        print(f"  {rid}: skipped, no total_score (incomplete DSRS)")
    out = out.loc[~no_score]

    blank = out["record_id"] == ""
    if blank.any():
        print(f"  {blank.sum()} severity row(s) skipped, blank record id")
    out = out.loc[~blank]

    dup = out["record_id"].duplicated(keep="first")
    for rid in out.loc[dup, "record_id"].unique():
        print(f"  duplicate severity id {rid}, first row kept")
    return out.loc[~dup].reset_index(drop=True)


def load_blacklist(path):
    if not path.is_file():
        raise SystemExit(f"blacklist file not found: {path}")
    ids = {normalize_id(line) for line in path.read_text().splitlines()}
    ids.discard("")
    print(f"Loaded {len(ids)} blacklisted id(s) from {path}")
    return ids


def compute_role_stats(path):
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return None
    if df.empty or "role" not in df.columns:
        return None
    counts = df["role"].value_counts()
    n = len(df)
    stats = {f"{r}_pct": 100 * int(counts.get(r, 0)) / n for r in TARGET_ROLES}
    stats["has_caregiver"] = int(counts.get("caregiver", 0)) > 0
    return stats


def select_visit(visits, timepoint, rid, site, tolerances, taken):
    have = ", ".join(f"V{v}" for v in sorted(visits))
    if timepoint == "T0":
        if 1 not in visits:
            print(f"  {rid} T0: skipped, no V1 transcript (has {have})")
            return None
        return 1, visits[1]

    if 1 not in visits:
        print(f"  {rid} {timepoint}: skipped, no V1 transcript to anchor on (has {have})")
        return None
    anchor = visits[1]["date"]
    if pd.isna(anchor):
        print(f"  {rid} {timepoint}: skipped, V1 filename has no parseable date ({'; '.join(visits[1]['files']['name'])})")
        return None
    months, tolerance = TIMEPOINT_MONTHS[timepoint], tolerances[timepoint]
    target = anchor + pd.DateOffset(months=months)
    margin = pd.Timedelta(days=tolerance * 30.4375)

    later = {v: info for v, info in visits.items() if v != 1}
    dated = {v: info for v, info in later.items() if pd.notna(info["date"])}
    in_window = {v: info for v, info in dated.items() if target - margin <= info["date"] <= target + margin}
    available = {v: info for v, info in in_window.items() if (site, rid, v) not in taken}
    if not available:
        if in_window:
            print(f"{rid} {timepoint}: skipped, only in-window visit(s) {', '.join(f'V{v}' for v in in_window)} already used for an earlier timepoint")
        elif dated:
            closest = min(dated, key=lambda v: abs(dated[v]["date"] - target))
            gap = (dated[closest]["date"] - anchor).days / 30.4375
            print(f"{rid} {timepoint}: skipped, no visit within {months}+/-{tolerance:g} months of V1 "
                  f"(closest V{closest} at {gap:.1f} months)")
        elif later:
            names = "; ".join(n for info in later.values() for n in info["files"]["name"])
            print(f"{rid} {timepoint}: skipped, later visits have no parseable date ({names})")
        else:
            print(f"{rid} {timepoint}: skipped, no transcripts after V1 (has {have})")
        return None
    best = min(available, key=lambda v: abs(available[v]["date"] - target))
    return best, available[best]


def print_visit_timing(people):
    gaps, out_of_order = {}, []
    for (site, rid), visits in people.items():
        anchor = visits.get(1, {}).get("date", pd.NaT)
        latest_label, latest_date = None, pd.NaT
        for visit in sorted(visits):
            date = visits[visit]["date"]
            if pd.isna(date):
                continue
            if pd.notna(latest_date) and date < latest_date:
                out_of_order.append(f"  {rid} V{visit} ({date.date()}) is dated before {latest_label} ({latest_date.date()})")
            else:
                latest_label, latest_date = f"V{visit}", date
            if visit != 1 and pd.notna(anchor):
                gaps.setdefault((site, visit), []).append((date - anchor).days / 30.4375)
    print("\n--- Visit timing, months after V1 ---")
    for (site, visit), values in sorted(gaps.items()):
        v = np.array(values)
        print(f"  {site} V{visit}: n {len(v)}, min {v.min():.1f}, 25% {np.percentile(v, 25):.1f}, "
              f"median {np.median(v):.1f}, 75% {np.percentile(v, 75):.1f}, max {v.max():.1f}")
    for line in sorted(out_of_order):
        print(line)


def build_dataset(
    transcripts_dir,
    severity_files,
    exclude_bilingual=False,
    require_caregiver=False,
    blacklist=frozenset(),
    tolerances=DEFAULT_TOLERANCES,
    extra_columns=(),
):
    transcripts = discover_transcripts(transcripts_dir)
    transcripts = transcripts[~transcripts["record_id"].isin(blacklist)].reset_index(drop=True)
    people = group_visits(transcripts)

    severity_sites = {site for site, _ in severity_files}
    for name in transcripts.loc[~transcripts["site"].isin(severity_sites), "name"]:
        print(f"  skipped, no severity file given for this site: {name}")

    feature_cols = [f"{r}_pct" for r in TARGET_ROLES]
    rows = []
    used_paths = set()
    severity_ids = set()
    taken = set()
    blacklisted_seen = set()
    for (site, timepoint), sev_path in sorted(severity_files.items()):
        print(f"\n--- Loading {site} {timepoint} ---")
        if timepoint != "T0" and timepoint not in TIMEPOINT_MONTHS:
            print(f"  skipped, no matching window defined for {timepoint}")
            continue
        sev = load_severity(sev_path, site, extra_columns)
        severity_ids.update((site, rid) for rid in sev["record_id"])

        for rec in sev.itertuples(index=False):
            if rec.record_id in blacklist:
                if rec.record_id not in blacklisted_seen:
                    print(f"  {rec.record_id}: excluded, on blacklist (all visits and timepoints)")
                blacklisted_seen.add(rec.record_id)
                continue

            visits = people.get((site, rec.record_id))
            if visits is None:
                print(f"  {rec.record_id} {timepoint}: skipped, no transcript with this id")
                continue

            selected = select_visit(visits, timepoint, rec.record_id, site, tolerances, taken)
            if selected is None:
                continue
            visit, info = selected
            grp = info["files"]

            if exclude_bilingual and grp["is_bilingual"].any():
                names = "; ".join(grp.loc[grp["is_bilingual"], "name"])
                print(f"  {rec.record_id} {timepoint}: excluded, bilingual ({names})")
                continue

            stats_list, used = [], []
            for path in grp["path"]:
                stats = compute_role_stats(path)
                if stats is None:
                    print(f"  {rec.record_id} {timepoint}: unreadable, empty, or no 'role' column: {path.name}")
                    continue
                if require_caregiver and not stats["has_caregiver"]:
                    print(f"  {rec.record_id} {timepoint}: excluded, no caregiver turns (dyad): {path.name}")
                    continue
                if stats["patient_pct"] == 0:
                    print(f"  {rec.record_id} {timepoint}: included with patient_pct = 0, no patient turns: {path.name}")
                stats_list.append(stats)
                used.append(path.name)
                used_paths.add(path)

            if not stats_list:
                continue
            if len(grp) > 1:
                print(f"  {rec.record_id} {timepoint}: {len(grp)} transcripts found for V{visit}, averaged")

            anchor = visits.get(1, {}).get("date", pd.NaT)
            months_from_v1 = (info["date"] - anchor).days / 30.4375 if pd.notna(info["date"]) and pd.notna(anchor) else np.nan
            if timepoint != "T0":
                taken.add((site, rec.record_id, visit))
            rows.append({
                "record_id": rec.record_id,
                "site": site,
                "timepoint": timepoint,
                "visit": f"V{visit}",
                "visit_date": info["date"].date().isoformat() if pd.notna(info["date"]) else "",
                "months_from_v1": round(months_from_v1, 1) if pd.notna(months_from_v1) else np.nan,
                "months_from_target": round(months_from_v1 - TIMEPOINT_MONTHS.get(timepoint, 0), 1) if pd.notna(months_from_v1) else np.nan,
                "total_score": float(rec.total_score),
                "severity_category": rec.severity_category,
                **{safe_column_name(c): getattr(rec, safe_column_name(c)) for c in extra_columns},
                **{c: float(np.mean([s[c] for s in stats_list])) for c in feature_cols},
                "is_bilingual": bool(grp["is_bilingual"].any()),
                "n_transcripts": len(stats_list),
                "transcript_files": "; ".join(used),
            })

    for rid in sorted(blacklist - blacklisted_seen):
        print(f"  blacklist id {rid} not found in any severity file given")

    print_visit_timing(people)

    print("\n--- Transcripts not matched to any DSRS timepoint ---")
    for _, row in transcripts[~transcripts["path"].isin(used_paths)].iterrows():
        reason = "no usable DSRS row for this id" if (row["site"], row["record_id"]) not in severity_ids else "not matched to a DSRS timepoint"
        print(f"  not used ({reason}): {row['name']}")

    columns = ["record_id", "site", "timepoint", "visit", "visit_date", "months_from_v1", "months_from_target", "total_score",
               "severity_category", *feature_cols, "is_bilingual", "n_transcripts", "transcript_files"]
    return pd.DataFrame(rows, columns=columns)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Join transcript role shares with DSRS severity.")
    ap.add_argument("--transcripts-dir", required=True,
                    help="Flat folder containing all transcript .csv files (all visits, both sites).")
    ap.add_argument("--severity-dir", required=True,
                    help="Folder of processed DSRS files named like 'Dartmouth DSRS_T0.xlsx' / 'NYU DSRS_T2.xlsx'.")
    ap.add_argument("--blacklist", default=None,
                    help="Text file with one record id per line to exclude (e.g. D01_0002), at every visit/timepoint.")
    ap.add_argument("--t2-tolerance", type=float, default=DEFAULT_TOLERANCES["T2"],
                    help="T2 matches the visit closest to 6 months after V1, within +/- this many months.")
    ap.add_argument("--t3-tolerance", type=float, default=DEFAULT_TOLERANCES["T3"],
                    help="T3 matches the visit closest to 12 months after V1, within +/- this many months.")
    ap.add_argument("--output", default="dataset.csv", help="Where to write the joined dataset.")
    ap.add_argument("--visits-output", default=None,
                    help="Also write one row per patient per visit (every transcript visit, with or without a DSRS) to this file.")
    ap.add_argument("--severity-extra-columns", nargs="+", default=[],
                    help="Columns of the processed DSRS files to carry into the dataset as sev_<name> (for example who the caregiver was).")
    ap.add_argument("--exclude-bilingual", action="store_true",
                    help="Drop rows whose matched transcript filename contains 'bilingual'.")
    ap.add_argument("--require-caregiver", action="store_true",
                    help="Drop transcripts with no caregiver-role turns (dyads).")
    args = ap.parse_args(argv)

    transcripts_dir = Path(args.transcripts_dir)
    if not transcripts_dir.is_dir():
        raise SystemExit(f"--transcripts-dir {transcripts_dir} does not exist or is not a folder.")
    severity_dir = Path(args.severity_dir)
    if not severity_dir.is_dir():
        raise SystemExit(f"--severity-dir {severity_dir} does not exist or is not a folder.")

    severity_files = find_severity_files(severity_dir)
    if not severity_files:
        raise SystemExit(f"No files named like '<Dartmouth|NYU> DSRS_T<n>' found in {severity_dir}.")
    for (site, timepoint), path in sorted(severity_files.items()):
        print(f"Severity file: {site} {timepoint}: {path.name}")

    print(f"Flags: exclude_bilingual={args.exclude_bilingual}, require_caregiver={args.require_caregiver}")
    df = build_dataset(transcripts_dir, severity_files,
                       exclude_bilingual=args.exclude_bilingual,
                       require_caregiver=args.require_caregiver,
                       blacklist=load_blacklist(Path(args.blacklist)) if args.blacklist else frozenset(),
                       tolerances={"T2": args.t2_tolerance, "T3": args.t3_tolerance},
                       extra_columns=tuple(args.severity_extra_columns))

    if df.empty:
        raise SystemExit("No usable records -- nothing written.")
    print("\n--- Matched T2/T3 rows ---")
    for _, r in df[df["timepoint"] != "T0"].sort_values(["site", "record_id", "timepoint"]).iterrows():
        print(f"  {r['site']} {r['record_id']} {r['timepoint']}: {r['visit']} {r['visit_date']}, "
              f"{r['months_from_v1']:.1f} months after V1, {r['months_from_target']:+.1f} from target")
    print(f"\n{len(df)} usable row(s):")
    print(pd.crosstab(df["site"], df["timepoint"]).to_string())
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

    if args.visits_output:
        visits = build_visit_table(transcripts_dir, load_blacklist(Path(args.blacklist)) if args.blacklist else frozenset(), args.exclude_bilingual)
        visits.to_csv(args.visits_output, index=False)
        per_patient = visits.groupby("record_id").size()
        print(f"\nWrote {args.visits_output}: {len(visits)} visit(s) from {len(per_patient)} patient(s). Patients with at least 1, 2, 3, 4 visits: "
              f"{int((per_patient >= 1).sum())}, {int((per_patient >= 2).sum())}, {int((per_patient >= 3).sum())}, {int((per_patient >= 4).sum())}")


if __name__ == "__main__":
    main()
