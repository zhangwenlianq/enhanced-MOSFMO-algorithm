"""Pareto utilities, constrained selection, and locked performance indicators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


Array = NDArray[np.float64]


def constraint_violation(g: Array) -> Array:
    values = np.asarray(g, dtype=float)
    if values.ndim != 2:
        raise ValueError("G must be a matrix")
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=float)
    return np.maximum(values, 0.0).sum(axis=1)


def feasible_mask(g: Array, tolerance: float = 0.0) -> NDArray[np.bool_]:
    return constraint_violation(g) <= tolerance


def nondominated_indices(f: Array) -> NDArray[np.int64]:
    values = np.asarray(f, dtype=float)
    if values.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    return np.asarray(NonDominatedSorting().do(values, only_non_dominated_front=True), dtype=np.int64)


def crowding_distance(f: Array) -> Array:
    values = np.asarray(f, dtype=float)
    n, m = values.shape
    if n == 0:
        return np.empty(0, dtype=float)
    if n <= 2:
        return np.full(n, np.inf)
    distance = np.zeros(n, dtype=float)
    for objective in range(m):
        order = np.argsort(values[:, objective], kind="stable")
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = values[order[-1], objective] - values[order[0], objective]
        if span > np.finfo(float).eps:
            interior = order[1:-1]
            distance[interior] += (
                values[order[2:], objective] - values[order[:-2], objective]
            ) / span
    return distance


def _stable_unique_indices(f: Array, x: Array) -> NDArray[np.int64]:
    if len(f) == 0:
        return np.empty(0, dtype=np.int64)
    joined = np.hstack([f, x])
    _, first = np.unique(joined, axis=0, return_index=True)
    return np.sort(first).astype(np.int64)


def update_archive(
    archive_x: Array,
    archive_f: Array,
    archive_g: Array,
    new_x: Array,
    new_f: Array,
    new_g: Array,
    capacity: int = 100,
) -> tuple[Array, Array, Array]:
    """Keep feasible nondominated points, with stable crowding truncation."""
    x = np.vstack([archive_x, new_x]) if len(archive_x) else np.asarray(new_x, dtype=float).copy()
    f = np.vstack([archive_f, new_f]) if len(archive_f) else np.asarray(new_f, dtype=float).copy()
    g = np.vstack([archive_g, new_g]) if len(archive_g) else np.asarray(new_g, dtype=float).copy()
    feasible = feasible_mask(g)
    x, f, g = x[feasible], f[feasible], g[feasible]
    if len(x) == 0:
        return x, f, g
    unique = _stable_unique_indices(f, x)
    x, f, g = x[unique], f[unique], g[unique]
    nd = nondominated_indices(f)
    x, f, g = x[nd], f[nd], g[nd]
    if len(x) > capacity:
        crowd = crowding_distance(f)
        order = np.lexsort(tuple(x[:, j] for j in reversed(range(x.shape[1]))) + (-crowd,))
        keep = order[:capacity]
        x, f, g = x[keep], f[keep], g[keep]
    return x, f, g


def constrained_better(
    f_new: Array,
    g_new: Array,
    f_old: Array,
    g_old: Array,
    weight: Array | None = None,
    ideal: Array | None = None,
) -> bool:
    cv_new = float(np.maximum(g_new, 0.0).sum()) if g_new.size else 0.0
    cv_old = float(np.maximum(g_old, 0.0).sum()) if g_old.size else 0.0
    if cv_new <= 0.0 < cv_old:
        return True
    if cv_old <= 0.0 < cv_new:
        return False
    if cv_new > 0.0 or cv_old > 0.0:
        return cv_new < cv_old
    new_dominates = np.all(f_new <= f_old) and np.any(f_new < f_old)
    old_dominates = np.all(f_old <= f_new) and np.any(f_old < f_new)
    if new_dominates:
        return True
    if old_dominates:
        return False
    if weight is None:
        return False
    z = np.minimum(f_new, f_old) if ideal is None else ideal
    w = np.maximum(np.asarray(weight, dtype=float), 1e-6)
    return float(np.max(w * np.abs(f_new - z))) < float(np.max(w * np.abs(f_old - z)))


def rank_and_crowding(f: Array, g: Array) -> tuple[NDArray[np.int64], Array]:
    n = len(f)
    ranks = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    crowd = np.full(n, -np.inf, dtype=float)
    cv = constraint_violation(g)
    feasible = np.flatnonzero(cv <= 0.0)
    if len(feasible):
        fronts = NonDominatedSorting().do(f[feasible])
        for rank, local in enumerate(fronts):
            indices = feasible[np.asarray(local, dtype=int)]
            ranks[indices] = rank
            crowd[indices] = crowding_distance(f[indices])
    infeasible = np.flatnonzero(cv > 0.0)
    if len(infeasible):
        order = infeasible[np.argsort(cv[infeasible], kind="stable")]
        base = int(ranks[feasible].max() + 1) if len(feasible) else 0
        ranks[order] = base + np.arange(len(order))
        crowd[order] = -cv[order]
    return ranks, crowd


def environmental_select(x: Array, f: Array, g: Array, size: int) -> tuple[Array, Array, Array]:
    ranks, crowd = rank_and_crowding(f, g)
    selected: list[int] = []
    for rank in np.unique(ranks):
        front = np.flatnonzero(ranks == rank)
        remaining = size - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front.tolist())
        else:
            order = np.argsort(-crowd[front], kind="stable")
            selected.extend(front[order[:remaining]].tolist())
            break
    idx = np.asarray(selected, dtype=int)
    return x[idx].copy(), f[idx].copy(), g[idx].copy()


def select_archive_leader(archive_x: Array, archive_f: Array, rng: np.random.Generator) -> Array:
    if len(archive_x) == 0:
        raise ValueError("Cannot select a leader from an empty archive")
    crowd = crowding_distance(archive_f)
    finite = crowd[np.isfinite(crowd)]
    edge_weight = (float(finite.max()) if len(finite) else 1.0) + 1.0
    weights = np.where(np.isfinite(crowd), np.maximum(crowd, 1e-12), edge_weight)
    probabilities = weights / weights.sum()
    return archive_x[int(rng.choice(len(archive_x), p=probabilities))].copy()


@dataclass(frozen=True)
class MetricContext:
    reference_front: Array
    ideal: Array
    nadir: Array
    normalized_reference: Array
    reference_point: Array
    reference_hv: float
    hv_indicator: HV
    igd_plus_indicator: IGDPlus

    @classmethod
    def from_reference_front(cls, reference_front: Array) -> "MetricContext":
        pf = np.asarray(reference_front, dtype=float)
        ideal = pf.min(axis=0)
        nadir = pf.max(axis=0)
        span = nadir - ideal
        if np.any(span <= np.finfo(float).eps):
            raise ValueError("Reference-front objective range is zero")
        normalized = (pf - ideal) / span
        reference_point = np.full(pf.shape[1], 1.1, dtype=float)
        hv_indicator = HV(ref_point=reference_point)
        reference_hv = float(hv_indicator(normalized))
        if not np.isfinite(reference_hv) or reference_hv <= 0.0:
            raise ValueError("Reference-front hypervolume is invalid")
        return cls(
            reference_front=pf.copy(), ideal=ideal, nadir=nadir,
            normalized_reference=normalized, reference_point=reference_point,
            reference_hv=reference_hv, hv_indicator=hv_indicator,
            igd_plus_indicator=IGDPlus(normalized),
        )

    def normalize(self, f: Array) -> Array:
        return (np.asarray(f, dtype=float) - self.ideal) / (self.nadir - self.ideal)

    def indicators(self, f: Array, g: Array) -> tuple[float, float]:
        feasible = feasible_mask(g)
        if not np.any(feasible):
            return 0.0, float("inf")
        values = np.asarray(f, dtype=float)[feasible]
        values = values[nondominated_indices(values)]
        normalized = self.normalize(values)
        within = np.all(normalized < self.reference_point, axis=1)
        hv = float(self.hv_indicator(normalized[within])) if np.any(within) else 0.0
        normalized_hv = float(np.clip(hv / self.reference_hv, 0.0, 1.0))
        igd_plus = float(self.igd_plus_indicator(normalized))
        return normalized_hv, igd_plus

