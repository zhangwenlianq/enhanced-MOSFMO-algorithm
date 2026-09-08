"""Candidate C001: energy-budgeted one-shot behavioral switching for MO-SFMO."""

from __future__ import annotations

import numpy as np

from research.baselines import (
    OptimizationResult,
    _finish,
    _initialize,
    _leader,
    _random_weights,
)
from research.benchmarks import Benchmark
from research.metrics import constrained_better, rank_and_crowding, update_archive


def _unit_rows(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, lengths, out=np.zeros_like(values), where=lengths > 1e-15)


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

    while evaluator.remaining:
        cycles += 1
        progress = evaluator.evaluations / budget
        count = min(n, evaluator.remaining)
        graze_step = 0.05 - (0.05 - 0.0005) * progress
        migration_step = 0.05 - (0.05 - 0.0005) * progress
        ranks, crowd = rank_and_crowding(f, g)
        order = np.lexsort((-crowd, ranks))
        predecessor = np.empty(n, dtype=int)
        predecessor[order[0]] = order[0]
        predecessor[order[1:]] = order[:-1]

        guides = np.asarray([_leader(archive, (x, f, g), rng) for _ in range(count)])
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
        accepted = np.zeros(count, dtype=bool)
        for i in range(count):
            if constrained_better(tf[i], tg[i], f[i], g[i], weights[i], ideal):
                previous_move[i] = trial[i] - x[i]
                x[i], f[i], g[i] = trial[i], tf[i], tg[i]
                accepted[i] = True
                stagnation[i] = 0
            else:
                stagnation[i] += 1
            if constrained_better(tf[i], tg[i], personal_f[i], personal_g[i], weights[i], ideal):
                personal_x[i], personal_f[i], personal_g[i] = trial[i], tf[i], tg[i]
        archive = update_archive(*archive, trial, tf, tg)
        recorder.record(archive)

    return _finish(
        algorithm, seed, evaluator, recorder, archive,
        {
            "candidate": "C001-one-shot",
            "cycles": cycles,
            "behavior_counts": behavior_counts,
            "population_size": n,
            "one_candidate_per_sheep_per_cycle": True,
        },
    )


__all__ = ["optimize"]
