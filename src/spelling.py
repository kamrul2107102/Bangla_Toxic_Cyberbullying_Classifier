from __future__ import annotations
import numpy as np

def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1); dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost)
    return int(dp[m, n])

def correct_token(token: str, vocabulary: list[str], max_distance: int = 2) -> str:
    if token in vocabulary:
        return token
    candidates = [(edit_distance(token, w), w) for w in vocabulary if abs(len(token)-len(w)) <= max_distance]
    if not candidates:
        return token
    distance, word = min(candidates)
    return word if distance <= max_distance else token
