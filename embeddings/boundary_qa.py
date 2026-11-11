"""
Forrest Leung
11/2/26 

Quality checks on llm_topic_labels.py's cached boundaries, run against existing cached visits.

To run:
python embeddings/boundary_qa.py "$DATA"
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from llm_topic_labels import DEFAULT_MERGE_COSINE_THRESHOLD, MIN_WORDS_FOR_BOUNDARY, STRIDE, WINDOW, cache_path
from load_data import load_roles, load_transcript

N_PERM = 2000
GAP_THRESHOLD = 2
SNIPPET_CONTEXT = 3                             # turns of context on each side
STRIDE_FRACTION = (WINDOW - STRIDE) / WINDOW    # the overlap's share of a window's span

# Same pool compute_boundaries() draws candidate boundaries from
def eligible_positions(t):
    return t.loc[t["word_count"] >= MIN_WORDS_FOR_BOUNDARY, "turn_index"].to_numpy()

def window_starts(n_turns):
    return list(range(0, max(n_turns - WINDOW + STRIDE, 1), STRIDE))

def gap_le_rate(idxs, threshold=GAP_THRESHOLD):
    idxs = sorted(idxs)
    if len(idxs) < 2:
        return 0.0, 0
    gaps = [idxs[i + 1] - idxs[i] for i in range(len(idxs) - 1)]
    close = sum(1 for g in gaps if g <= threshold)
    return close / len(gaps), close

def null_gap_rate(eligible, n_boundaries, n_perm=N_PERM, threshold=GAP_THRESHOLD, seed=0):
    rng = np.random.default_rng(seed)
    rates = np.empty(n_perm)
    for i in range(n_perm):
        sample = rng.choice(eligible, size=min(n_boundaries, len(eligible)), replace=False)
        rates[i], _ = gap_le_rate(sample, threshold)
    return rates

# Window edge flags for boundaries that fall in the overlap zone of later windows
def window_edge_flags(idxs, n_turns):
    starts = [s for s in window_starts(n_turns) if s > 0]
    flags = {}
    for idx in idxs:
        flags[idx] = any(s <= idx < s + (WINDOW - STRIDE) for s in starts)
    return flags

def analyze_visit(path, roles, outputs_dir):
    vid_cache = cache_path(outputs_dir, _visit_id(path))
    if not os.path.exists(vid_cache):
        return None
    with open(vid_cache) as f:
        data = json.load(f)
    labels = {int(k): v for k, v in data["labels"].items()}
    idxs = sorted(labels.keys())

    t = load_transcript(path, roles)
    n_turns = len(t)
    eligible = eligible_positions(t)

    obs_rate, obs_close = gap_le_rate(idxs)
    null_rates = null_gap_rate(eligible, len(idxs))
    p_value = float((np.sum(null_rates >= obs_rate) + 1) / (N_PERM + 1))

    edge_flags = window_edge_flags(idxs, n_turns)
    gaps_pairs = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1) if idxs[i + 1] - idxs[i] <= GAP_THRESHOLD]
    short_gap_turns = sorted(set(a for a, b in gaps_pairs) | set(b for a, b in gaps_pairs))
    edge_adjacent_short_gap = sum(1 for x in short_gap_turns if edge_flags.get(x))
    edge_adjacent_all = sum(1 for x in idxs if edge_flags.get(x))

    role_by_turn = t.set_index("turn_index")["role"]
    all_roles = role_by_turn.reindex(idxs).value_counts()
    short_gap_roles = role_by_turn.reindex(short_gap_turns).value_counts()

    return {
        "n_turns": n_turns, "n_boundaries": len(idxs),
        "observed_gap_le2_rate": obs_rate, "observed_gap_le2_count": obs_close,
        "null_mean": float(null_rates.mean()), "null_std": float(null_rates.std()),
        "p_value": p_value,
        "n_short_gap_turns": len(short_gap_turns),
        "n_short_gap_turns_window_edge_adjacent": edge_adjacent_short_gap,
        "pct_short_gap_window_edge_adjacent": edge_adjacent_short_gap / max(len(short_gap_turns), 1),
        "pct_all_boundaries_window_edge_adjacent": edge_adjacent_all / max(len(idxs), 1),
        "role_counts_all_boundaries": all_roles.to_dict(),
        "role_counts_short_gap_boundaries": short_gap_roles.to_dict(),
        "idxs": idxs, "labels": labels, "short_gap_turns": short_gap_turns,
    }

def _visit_id(path):
    from load_data import key
    return key(path)

# Export boundary snippets for manual review
def export_snippets(path, t, idxs, short_gap_turns, out_f, n=3):
    picks = short_gap_turns[:2] + [i for i in idxs if i not in short_gap_turns][:1]
    picks = picks[:n]
    out_f.write(f"=== {os.path.basename(path)} ===\n")
    for idx in picks:
        lo, hi = max(idx - SNIPPET_CONTEXT, t["turn_index"].min()), idx + SNIPPET_CONTEXT
        window = t[(t["turn_index"] >= lo) & (t["turn_index"] <= hi)]
        tag = "SHORT-GAP" if idx in short_gap_turns else "normal"
        out_f.write(f"\n-- boundary at turn {idx} [{tag}] --\n")
        for _, r in window.iterrows():
            marker = " <== BOUNDARY" if r["turn_index"] == idx else ""
            out_f.write(f"  [{r['turn_index']}] {r['role'].upper()}: {r['text']}{marker}\n")
    out_f.write("\n")

# Merge-collapse zone report for sanity checking the label-cosine merge
def merge_collapse_zone_report(idx_df, roles, outputs_dir, cosine_threshold):
    from sentence_transformers import SentenceTransformer
    from llm_topic_labels import merge_boundaries
    merge_model = SentenceTransformer("FremyCompany/BioLORD-2023", device="cpu")

    in_zone, out_zone, n_visits = 0, 0, 0
    for _, row in idx_df.iterrows():
        vid = _visit_id(row["path"])
        vc = cache_path(outputs_dir, vid)
        if not os.path.exists(vc):
            continue
        with open(vc) as f:
            data = json.load(f)
        labels = {int(k): v for k, v in data["labels"].items()}
        if len(labels) < 2:
            continue
        t = load_transcript(row["path"], roles)
        merged = merge_boundaries(labels, merge_model, cosine_threshold)        # dedup_stitching off by default
        dropped = sorted(set(labels) - set(merged))
        if not dropped:
            n_visits += 1
            continue
        flags = window_edge_flags(dropped, len(t))
        in_zone += sum(1 for v in flags.values() if v)
        out_zone += sum(1 for v in flags.values() if not v)
        n_visits += 1

    total = in_zone + out_zone
    print(f"\nmerge-collapse zone report ({n_visits} visits, cosine_threshold={cosine_threshold}):")
    print(f"  pairs collapsed by the label-cosine merge, in a window overlap zone:     {in_zone}")
    print(f"  pairs collapsed by the label-cosine merge, outside a window overlap zone: {out_zone}")
    if total:
        print(f"  in-zone share: {in_zone / total:.1%} "
              f"(for reference, the overlap zone is {STRIDE_FRACTION:.1%} of a typical window's span)")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--outputs-dir", default="outputs/topics")
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--merge-zone-report", action="store_true",
                    help="also run the merge-collapse-by-zone sanity check (loads the "
                         "embedding model; skips the per-visit gap/role checks)")
    args = ap.parse_args()

    from load_data import load_dataset
    idx_df = load_dataset(args.data_dir, verbose=False)
    roles = load_roles(args.data_dir)

    if args.merge_zone_report:
        merge_collapse_zone_report(idx_df, roles, args.outputs_dir, DEFAULT_MERGE_COSINE_THRESHOLD)
        return

    snippet_path = os.path.join(args.data_dir, "derived", "boundary_snippets.txt")
    os.makedirs(os.path.dirname(snippet_path), exist_ok=True)

    rows = []
    with open(snippet_path, "w") as snippet_f:
        for _, row in idx_df.iterrows():
            result = analyze_visit(row["path"], roles, args.outputs_dir)
            if result is None:
                continue
            t = load_transcript(row["path"], roles)
            export_snippets(row["path"], t, result["idxs"], result["short_gap_turns"], snippet_f)
            rows.append({k: v for k, v in result.items() if k not in ("idxs", "labels", "short_gap_turns")})
            print(f"{os.path.basename(row['path'])}: n_boundaries={result['n_boundaries']} "
                  f"observed_gap<=2={result['observed_gap_le2_rate']:.1%} "
                  f"null_mean={result['null_mean']:.1%} p={result['p_value']:.3f} "
                  f"window_edge_adjacent(short-gap)={result['pct_short_gap_window_edge_adjacent']:.1%} "
                  f"window_edge_adjacent(all)={result['pct_all_boundaries_window_edge_adjacent']:.1%}")

    if not rows:
        print("no cached visits found yet")
        return

    df = pd.DataFrame(rows)
    os.makedirs(args.results_dir, exist_ok=True)
    df.to_csv(os.path.join(args.results_dir, "boundary_qa.csv"), index=False)
    print(f"\nwrote {len(df)}-visit QA table -> {args.results_dir}/boundary_qa.csv")
    print(f"wrote snippets (raw text, local only) -> {snippet_path}")

    all_role = {}
    short_role = {}
    for r in rows:
        for k, v in r["role_counts_all_boundaries"].items():
            all_role[k] = all_role.get(k, 0) + v
        for k, v in r["role_counts_short_gap_boundaries"].items():
            short_role[k] = short_role.get(k, 0) + v
    print("\nrole breakdown, ALL boundaries:", all_role)
    print("role breakdown, SHORT-GAP boundaries:", short_role)

if __name__ == "__main__":
    main()