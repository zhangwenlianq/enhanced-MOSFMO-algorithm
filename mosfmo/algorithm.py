"""Candidate C003: vectorized one-shot SFMO with a shared flock scent field."""

from __future__ import annotations

import numpy as np

from research.baselines import (
    OptimizationResult,
    _finish,
    _initialize,
    _random_weights,
)
from research.benchmarks import Benchmark
from research.metrics import (
    constraint_violation,
    crowding_distance,
    rank_and_crowding,
    update_archive,
)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, lengths, out=np.zeros_like(values), where=lengths > 1e-15)


def _better_mask(
    f_new: np.ndarray,
    g_new: np.ndarray,
    f_old: np.ndarray,
    g_old: np.ndarray,
    weights: np.ndarray,
    ideal: np.ndarray,
) -> np.ndarray:
    """Vectorized equivalent of the locked constrained scalar replacement."""
    cv_new = np.maximum(g_new, 0.0).sum(axis=1) if g_new.shape[1] else np.zeros(len(f_new))
    cv_old = np.maximum(g_old, 0.0).sum(axis=1) if g_old.shape[1] else np.zeros(len(f_old))
    new_feasible = cv_new <= 0.0
    old_feasible = cv_old <= 0.0
    better = new_feasible & ~old_feasible
    both_infeasible = ~new_feasible & ~old_feasible
    better[both_infeasible] = cv_new[both_infeasible] < cv_old[both_infeasible]
    both_feasible = new_feasible & old_feasible
    if np.any(both_feasible):
        idx = np.flatnonzero(both_feasible)
        fn, fo = f_new[idx], f_old[idx]
        new_dominates = np.all(fn <= fo, axis=1) & np.any(fn < fo, axis=1)
        old_dominates = np.all(fo <= fn, axis=1) & np.any(fo < fn, axis=1)
        undecided = ~(new_dominates | old_dominates)
        local = new_dominates.copy()
        if np.any(undecided):
            u = idx[undecided]
            w = np.maximum(weights[u], 1e-6)
            new_score = np.max(w * np.abs(f_new[u] - ideal), axis=1)
            old_score = np.max(w * np.abs(f_old[u] - ideal), axis=1)
            local[undecided] = new_score < old_score
        better[idx] = local
    return better


def _sample_scent_guides(
    archive: tuple[np.ndarray, np.ndarray, np.ndarray],
    population: tuple[np.ndarray, np.ndarray, np.ndarray],
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    """Sample all social cues from one flock-wide density perception."""
    ax, af, _ = archive
    if len(ax):
        crowd = crowding_distance(af)
        finite = crowd[np.isfinite(crowd)]
        edge_weight = (float(finite.max()) if len(finite) else 1.0) + 1.0
        scent = np.where(np.isfinite(crowd), np.maximum(crowd, 1e-12), edge_weight)
        chosen = rng.choice(len(ax), size=count, p=scent / scent.sum())
        return ax[np.asarray(chosen, dtype=int)].copy()
    px, _, pg = population
    cv = constraint_violation(pg)
    pool = np.flatnonzero(cv == cv.min())
    return px[rng.choice(pool, size=count)].copy()


def optimize(
    problem: Benchmark,
    initial_x: np.ndarray,
    seed: int,
    budget: int,
) -> OptimizationResult:
    """Evaluate exactly one behavior-generated candidate per active sheep and cycle."""
    algorithm = "ash_mosfmo"
    rng, evaluator, recorder, x, f, g, archive = _initialize(
        algorithm, problem, initial_x, seed, budget
    )
    n = len(x)
    weights = _random_weights(rng, n, problem.n_obj)
    personal_x, personal_f, personal_g = x.copy(), f.copy(), g.copy()
    stagnation = np.zeros(n, dtype=int)
    previous_move = np.zeros_like(x)
    behavior_counts = {"graze": 0, "migrate": 0, "regroup": 0}
    cycles = 0
    social_interval = 5
    pending_x = x.copy()
    pending_f = f.copy()
    pending_g = g.copy()
    pending_valid = np.zeros(n, dtype=bool)
    ranks, crowd = rank_and_crowding(f, g)
    order = np.lexsort((-crowd, ranks))
    predecessor = np.empty(n, dtype=int)
    predecessor[order[0]] = order[0]
    predecessor[order[1:]] = order[:-1]
    guides = _sample_scent_guides(archive, (x, f, g), rng, n)

    while evaluator.remaining:
        cycles += 1
        progress = evaluator.evaluations / budget
        count = min(n, evaluator.remaining)
        graze_step = 0.05 - (0.05 - 0.0005) * progress
        migration_step = 0.05 - (0.05 - 0.0005) * progress
        forced_regroup = stagnation[:count] >= 5
        graze_probability = 0.65 * (1.0 - progress) + 0.20
        choices = rng.random(count)
        grazing = (~forced_regroup) & (choices < graze_probability)
        migrating = (~forced_regroup) & (~grazing)
        regrouping = forced_regroup

        trial = x[:count].copy()
        if np.any(grazing):
            idx = np.flatnonzero(grazing)
            center = personal_x[idx]
            trial[idx] = center + graze_step * rng.uniform(-1.0, 1.0, size=center.shape)
            behavior_counts["graze"] += len(idx)

        if np.any(migrating):
            idx = np.flatnonzero(migrating)
            yearning = _unit_rows(personal_x[idx] - x[idx])
            leading = _unit_rows(previous_move[predecessor[idx]])
            following = _unit_rows(x[predecessor[idx]] - x[idx])
            archive_direction = _unit_rows(guides[idx] - x[idx])
            direction = _unit_rows(
                0.35 * yearning + 0.20 * leading + 0.20 * following + 0.25 * archive_direction
            )
            trial[idx] = x[idx] + migration_step * rng.random((len(idx), 1)) * direction
            behavior_counts["migrate"] += len(idx)

        if np.any(regrouping):
            idx = np.flatnonzero(regrouping)
            social = 0.5 * rng.random((len(idx), 1)) * (guides[idx] - x[idx])
            compensation = 2.0 * graze_step * rng.uniform(-1.0, 1.0, size=(len(idx), problem.n_var))
            trial[idx] = x[idx] + social + compensation
            behavior_counts["regroup"] += len(idx)

        trial = np.clip(trial, 0.0, 1.0)
        tf, tg = evaluator.evaluate(trial)
        ideal = np.minimum(f.min(axis=0), tf.min(axis=0))
        accepted = _better_mask(tf, tg, f[:count], g[:count], weights[:count], ideal)
        rejected = ~accepted
        previous_move[:count][accepted] = trial[accepted] - x[:count][accepted]
        x[:count][accepted] = trial[accepted]
        f[:count][accepted] = tf[accepted]
        g[:count][accepted] = tg[accepted]
        stagnation[:count][accepted] = 0
        stagnation[:count][rejected] += 1

        personal_accept = _better_mask(
            tf, tg, personal_f[:count], personal_g[:count], weights[:count], ideal
        )
        personal_x[:count][personal_accept] = trial[personal_accept]
        personal_f[:count][personal_accept] = tf[personal_accept]
        personal_g[:count][personal_accept] = tg[personal_accept]

        pending_accept = ~pending_valid[:count]
        compare = np.flatnonzero(pending_valid[:count])
        if len(compare):
            pending_accept[compare] = _better_mask(
                tf[compare],
                tg[compare],
                pending_f[compare],
                pending_g[compare],
                weights[compare],
                ideal,
            )
        pending_x[:count][pending_accept] = trial[pending_accept]
        pending_f[:count][pending_accept] = tf[pending_accept]
        pending_g[:count][pending_accept] = tg[pending_accept]
        pending_valid[:count][pending_accept] = True

        social_pause = cycles % social_interval == 0 or evaluator.remaining == 0
        if social_pause:
            share = np.flatnonzero(pending_valid)
            if len(share):
                archive = update_archive(
                    *archive, pending_x[share], pending_f[share], pending_g[share]
                )
            pending_valid[:] = False
            ranks, crowd = rank_and_crowding(f, g)
            order = np.lexsort((-crowd, ranks))
            predecessor[order[0]] = order[0]
            predecessor[order[1:]] = order[:-1]
            guides = _sample_scent_guides(archive, (x, f, g), rng, n)
            recorder.record(archive)

    return _finish(
        algorithm, seed, evaluator, recorder, archive,
        {
            "candidate": "C003-collective-scent-field",
            "cycles": cycles,
            "behavior_counts": behavior_counts,
            "population_size": n,
            "one_candidate_per_sheep_per_cycle": True,
            "social_interval_cycles": social_interval,
            "per_sheep_rumination_buffer": True,
            "vectorized_replacement": True,
            "shared_scent_field": True,
        },
    )


__all__ = ["optimize"]
