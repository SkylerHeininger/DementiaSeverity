"""
Forrest Leung
11/2/26

This module implements embedding-based topic-boundary detection using a TextTiling-style approach (Hearst 1997).
It computes cosine similarity between the mean embeddings of adjacent turns, calculates depth scores, and
selects boundaries based on a threshold and local maxima criteria.
"""
import numpy as np

CONTEXT_K = 3
CUTOFF_SIGMA = 1.0
MIN_GAP_TURNS = 2

# Constants for the topic segmentation algorithm
def similarity_curve(E, k=CONTEXT_K):
    n = len(E)
    sim = np.full(max(n - 1, 0), np.nan)
    for i in range(n - 1):
        left = E[max(0, i - k + 1):i + 1].mean(0)
        right = E[i + 1:i + 1 + k].mean(0)
        ln, rn = np.linalg.norm(left), np.linalg.norm(right)
        if ln > 0 and rn > 0:
            sim[i] = float(left @ right) / (ln * rn)
    return sim

# Compute depth scores for the similarity curve
def depth_scores(sim):
    n = len(sim)
    depth = np.zeros(n)
    for i in range(n):
        if np.isnan(sim[i]):
            continue
        l = i
        while l > 0 and not np.isnan(sim[l - 1]) and sim[l - 1] >= sim[l]:
            l -= 1
        r = i
        while r < n - 1 and not np.isnan(sim[r + 1]) and sim[r + 1] >= sim[r]:
            r += 1
        depth[i] = max((sim[l] - sim[i]) + (sim[r] - sim[i]), 0.0)
    return depth

# Select topic boundaries based on depth scores, threshold, and local maxima criteria
def select_boundaries(depth, cutoff_sigma=CUTOFF_SIGMA, min_gap=MIN_GAP_TURNS):
    valid = depth[~np.isnan(depth)] if len(depth) else depth
    if len(valid) < 2 or valid.std() == 0:
        return np.array([], dtype=int)
    threshold = valid.mean() + cutoff_sigma * valid.std()
    chosen = []
    for i in range(len(depth)):
        if depth[i] <= threshold:
            continue
        if i > 0 and depth[i - 1] > depth[i]:
            continue
        if i < len(depth) - 1 and depth[i + 1] > depth[i]:
            continue
        if chosen and i - chosen[-1] < min_gap:
            if depth[i] <= depth[chosen[-1]]:
                continue
            chosen.pop()
        chosen.append(i)
    return np.array(chosen, dtype=int)

# Compute topic boundaries for a given set of turns and embeddings
def embedding_boundaries(t, emb, min_words, k=CONTEXT_K, cutoff_sigma=CUTOFF_SIGMA, min_gap=MIN_GAP_TURNS):
    sub_idx = np.where((t["word_count"] >= min_words).to_numpy())[0]
    flags = np.zeros(len(t), dtype=int)
    if len(sub_idx) < 2 * k + 1:
        return flags
    sim = similarity_curve(emb[sub_idx], k)
    depth = depth_scores(sim)
    for gap in select_boundaries(depth, cutoff_sigma, min_gap):
        flags[sub_idx[gap + 1]] = 1
    return flags