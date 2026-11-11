"""
Forrest Leung
10/26/26

Check and see if a longitudinal analysis is feasible on csv files.

To run locally + get aggregate counts:
python check_longitudinal.py example.csv
"""
import re
import sys
from pathlib import Path

import pandas as pd

paths = sys.argv[1:] or sys.exit("usage: python check_longitudinal.py <severity csv> [more csvs ...]")
frames = []
for p in paths:
    if Path(p).name.startswith("~$"):
        continue  # Excel lock file left behind while a workbook is open
    if Path(p).suffix.lower() in {".xlsx", ".xls"}:
        f = pd.read_excel(p)  # first sheet
    else:
        f = pd.read_csv(p, sep=None, engine="python")
    f["source_file"] = Path(p).name
    frames.append(f)
df = pd.concat(frames, ignore_index=True)
print("rows per file:")
print(df["source_file"].value_counts().sort_index(), "\n")
pid, ev = "record_id_clean", "Event Name"
if ev not in df or df[ev].isna().all():
    ev = "source_file"   # fall back to the file name as the timepoint

scored = df.dropna(subset=["total_score"])
usable = scored.dropna(subset=["matched_transcript"])
print(f"rows: {len(df)} | with total_score: {len(scored)} | with score AND transcript: {len(usable)}")
print(f"patients: {df[pid].nunique()} | with >=1 usable visit: {usable[pid].nunique()}")

print("\nevents (all rows / usable rows):")
print(pd.concat([df[ev].value_counts().rename("all"),
                 usable[ev].value_counts().rename("usable")], axis=1).fillna(0).astype(int))

visits = usable.groupby(pid).size()
print("\nusable visits per patient:")
print(visits.value_counts().sort_index().rename_axis("n_visits").rename("n_patients"))
print(f"patients with >=2 usable visits: {(visits >= 2).sum()}")

# Analyze longitudinal changes in total_score and severity_category for patients with multiple usable visits.
def event_order(name):
    m = re.search(r"\bT(\d+)\b", str(name))
    return int(m.group(1)) if m else 99

multi = usable[usable[pid].isin(visits[visits >= 2].index)].copy()
multi["event_order"] = multi[ev].map(event_order)
if (multi["event_order"] == 99).any():
    print("WARNING: some event names have no T<number>; their order is unknown")
multi = multi.sort_values([pid, "event_order"])
change = multi.groupby(pid)["total_score"].agg(lambda s: s.iloc[-1] - s.iloc[0])
if len(change):
    print("\nfirst-to-last change in total_score:")
    print(change.describe().round(2))
    print(f"patients whose score changed at all: {(change != 0).sum()} / {len(change)}")
    print("\nseverity_category transitions (first -> last):")
    first_last = multi.groupby(pid)["severity_category"].agg(["first", "last"])
    print(pd.crosstab(first_last["first"], first_last["last"]))

print("\ntotal_score distribution (all usable visits):")
print(usable["total_score"].describe().round(2))
print(usable["severity_category"].value_counts())