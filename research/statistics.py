"""Locked statistical summaries for final multi-algorithm comparisons."""

from __future__ import annotations

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def friedman_test(samples: dict[str, list[float]]) -> dict[str, float | int | None]:
    if len(samples) < 3:
        return {"statistic": None, "p_value": None, "blocks": 0}
    arrays = [np.asarray(value, dtype=float) for value in samples.values()]
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        return {"statistic": None, "p_value": None, "blocks": 0}
    result = friedmanchisquare(*arrays)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "blocks": len(arrays[0])}


def paired_wilcoxon(a: list[float], b: list[float]) -> dict[str, float | int | None]:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape or len(x) < 2:
        return {"statistic": None, "p_value": None, "pairs": 0}
    if np.allclose(x, y, rtol=0.0, atol=0.0):
        return {"statistic": 0.0, "p_value": 1.0, "pairs": len(x)}
    result = wilcoxon(x, y, alternative="two-sided", zero_method="pratt")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "pairs": len(x)}


def hierarchical_bootstrap_difference(
    by_problem_a: list[list[float]],
    by_problem_b: list[list[float]],
    *,
    seed: int = 424242,
    repetitions: int = 10000,
) -> dict[str, float]:
    if len(by_problem_a) != len(by_problem_b) or not by_problem_a:
        raise ValueError("Paired non-empty problem blocks are required")
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    n_problem = len(by_problem_a)
    for k in range(repetitions):
        selected = rng.integers(0, n_problem, size=n_problem)
        deltas = []
        for index in selected:
            a = np.asarray(by_problem_a[index], dtype=float)
            b = np.asarray(by_problem_b[index], dtype=float)
            if a.shape != b.shape:
                raise ValueError("Seed blocks must be paired")
            seeds = rng.integers(0, len(a), size=len(a))
            deltas.append(float(np.median(a[seeds] - b[seeds])))
        draws[k] = float(np.mean(deltas))
    return {
        "estimate": float(np.mean([np.median(np.asarray(a) - np.asarray(b)) for a, b in zip(by_problem_a, by_problem_b, strict=True)])),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }
