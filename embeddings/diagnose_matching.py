"""
Forrest Leung
11/2/26 

Prints counts, month summaries, and masked patterns for unmatched severity rows and transcripts.

To run:
python embeddings/diagnose_matching.py "$DATA"
python embeddings/diagnose_matching.py "$DATA" --site NYU
"""
import argparse
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from load_data import (TIMEPOINT_MONTHS, WINDOW_MONTHS, build_index, find_transcripts, key,
                       load_severity, norm_id)

def mask(s):
    return re.sub(r"\d", "9", re.sub(r"[A-Za-z]", "A", str(s)))

# Folder names are kept readable, except anything shaped like a patient ID or a date.
def redact_dir(d):
    d = re.sub(r"[A-Za-z]\d{2}[-_ ]?\d{3,4}", "<ID>", d)
    return re.sub(r"\d{1,2}\s*[A-Za-z]{3,9}\s*\d{2,4}", "<DATE>", d)

def top_patterns(values, n=8):
    c = Counter(mask(v) for v in values if pd.notna(v) and str(v).strip())
    return "\n".join(f"    {cnt:4d}  {pat}" for pat, cnt in c.most_common(n)) or "    (none)"

def site_of(source_file):
    s = str(source_file).lower()
    return "NYU" if "nyu" in s else "Dartmouth" if "dartmouth" in s else "?"

# Determine the location of transcripts given a list of keys and all available files.
def where_is(keys, all_files):
    hits = [h for k in keys for h in all_files.get(k, [])]
    if not hits:
        return "not found anywhere"
    parts = " | ".join(hits).lower()
    exts = {Path(h).suffix.lower() for h in hits}
    if "bad_diarization" in parts:
        return "in bad_diarization"
    if "bilingual" in parts:
        return "in V1 Bilingual"
    if ".csv" not in exts:
        return "only as " + "/".join(sorted(exts))
    csv_dir = next(os.path.dirname(h) for h in hits if h.lower().endswith(".csv"))
    return "CSV exists in: " + redact_dir(csv_dir)

def month_summary(g):
    m = g["months"].dropna()
    if m.empty:
        return "no dates"
    return (f"n={len(m)}  median={m.median():.1f}  IQR={m.quantile(.25):.1f}-{m.quantile(.75):.1f}"
            f"  range={m.min():.1f}-{m.max():.1f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--site", choices=["NYU", "Dartmouth"])
    args = ap.parse_args()

    print("=== running the normal matching ===")
    found = build_index(args.data_dir, verbose=True)
    found["site_sev"] = found["source_file"].map(site_of)

    # The team's own single-file pairings show how far after baseline each timepoint's transcript is.
    team = found[found["match_method"] == "matched_transcript"]
    print("\n=== months since first visit, for transcripts the team paired with each timepoint ===")
    print(f"(matching assumes targets {TIMEPOINT_MONTHS} months, within ±{WINDOW_MONTHS})")
    for (site, tp), g in team.groupby(["site", "timepoint"]):
        print(f"  {site:10s} T{int(tp)}: {month_summary(g)}")
    print("months since first visit, all matched pairs by method:")
    for (method, tp), g in found.groupby(["match_method", "timepoint"]):
        print(f"  {method:36s} T{int(tp)}: {month_summary(g)}")

    sev = load_severity(args.data_dir)
    sev = sev.dropna(subset=["total_score"]).copy()
    sev["site"] = sev["source_file"].map(site_of)
    sev["pid_norm"] = sev["record_id_clean"].map(norm_id)
    tr = find_transcripts(args.data_dir)

    matched_rows = set(zip(found["record_id_clean"], found["Event Name"]))
    un = sev[[(r, e) not in matched_rows for r, e in zip(sev["record_id_clean"], sev["Event Name"])]]
    un_tr = tr[~tr["key"].isin(found["key"])]
    if args.site:
        un, un_tr = un[un["site"] == args.site], un_tr[un_tr["site"] == args.site]

    print(f"\n=== unmatched scored severity rows: {len(un)} ===")
    print(pd.crosstab(un["site"], un["timepoint"].fillna(-1).astype(int)
                      .replace(-1, "no T#").rename("timepoint")).to_string())

    all_files = {}
    for root, _, files in os.walk(args.data_dir):
        if os.sep + "derived" in root:
            continue
        for f in files:
            all_files.setdefault(key(f), []).append(os.path.relpath(os.path.join(root, f), args.data_dir))
    named = un[un["cand_keys"].str.len() > 0]
    print(f"\n--- {len(named)} name a transcript in matched_transcript; where is that file? ---")
    print(named["cand_keys"].map(lambda k: where_is(k, all_files)).value_counts().to_string())

    pids_with_files = set(tr["pid_norm"].dropna())
    has_any = un["pid_norm"].isin(pids_with_files)
    print(f"\n--- {has_any.sum()} of {len(un)} unmatched rows are for patients who have "
          f"some transcript CSV (no transcript fit the timepoint's date window) ---")
    if has_any.any():
        # for each such row: how far is that patient's nearest transcript from the target month?
        gaps = []
        for _, r in un[has_any].iterrows():
            target = TIMEPOINT_MONTHS.get(int(r["timepoint"])) if pd.notna(r["timepoint"]) else None
            m = tr.loc[tr["pid_norm"] == r["pid_norm"], "months"].dropna()
            if target is not None and len(m):
                gaps.append({"timepoint": f"T{int(r['timepoint'])}",
                             "nearest_gap": (m - target).abs().min(),
                             "latest_month": m.max()})
        if gaps:
            g = pd.DataFrame(gaps)
            print("distance (months) from the target to that patient's nearest transcript:")
            print(g.groupby("timepoint")["nearest_gap"].describe()[["count", "50%", "min", "max"]]
                  .round(1).to_string())
            print("how far that patient's transcripts extend (latest month):")
            print(g.groupby("timepoint")["latest_month"].describe()[["50%", "min", "max"]]
                  .round(1).to_string())

    print(f"\n=== unmatched transcript CSVs: {len(un_tr)} ===")
    print("file name patterns:")
    print(top_patterns(un_tr["path"].map(lambda p: Path(p).name)))
    print(f"no patient ID parsed: {un_tr['pid_norm'].isna().sum()} | "
          f"no visit number: {un_tr['visit'].isna().sum()} | no date: {un_tr['date'].isna().sum()}")
    print("folders:")
    print(un_tr["path"].map(lambda p: redact_dir(os.path.relpath(os.path.dirname(p), args.data_dir)))
          .value_counts().to_string())
    print("visit numbers:", dict(un_tr["visit"].value_counts().sort_index()))

if __name__ == "__main__":
    main()