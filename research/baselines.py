"""Frozen baseline optimizers and the shared strict-NFE execution machinery."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from research.benchmarks import Benchmark
from research.metrics import (
    constrained_better,
    constraint_violation,
    environmental_select,
    rank_and_crowding,
    select_archive_leader,
    update_archive,
)


Array = NDArray[np.float64]


@dataclass
class Checkpoint:
    evaluations: int
    elapsed_seconds: float
    archive_x: Array
    archive_f: Array
    archive_g: Array


@dataclass
class OptimizationResult:
    algorithm: str
    seed: int
    evaluations: int
    feasible_evaluations: int
    elapsed_seconds: float
    archive_x: Array
    archive_f: Array
    archive_g: Array
    checkpoints: list[Checkpoint]
    metadata: dict[str, object] = field(default_factory=dict)


class BudgetedEvaluator:
    def __init__(self, problem: Benchmark, budget: int) -> None:
        self.problem = problem
        self.budget = int(budget)
        self.evaluations = 0
        self.feasible_evaluations = 0
        self.started_at = time.perf_counter()

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    def evaluate(self, x: Array) -> tuple[Array, Array]:
        values = np.atleast_2d(np.asarray(x, dtype=float))
        if len(values) > self.remaining:
            raise RuntimeError("Objective evaluation would exceed the strict NFE budget")
        f, g = self.problem.evaluate(values)
        self.evaluations += len(values)
        cv = constraint_violation(g)
        self.feasible_evaluations += int(np.count_nonzero(cv <= 0.0))
        return f, g

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at


class TraceRecorder:
    def __init__(self, evaluator: BudgetedEvaluator, interval: int = 500) -> None:
        self.evaluator = evaluator
        self.interval = interval
        self.next_boundary = interval
        self.checkpoints: list[Checkpoint] = []

    def record(self, archive: tuple[Array, Array, Array], force: bool = False) -> None:
        crossed = self.evaluator.evaluations >= self.next_boundary
        if not crossed and not force:
            return
        x, f, g = archive
        if crossed:
            while self.next_boundary <= self.evaluator.evaluations:
                self.next_boundary += self.interval
        if self.checkpoints and self.checkpoints[-1].evaluations == self.evaluator.evaluations:
            return
        self.checkpoints.append(
            Checkpoint(
                evaluations=self.evaluator.evaluations,
                elapsed_seconds=self.evaluator.elapsed(),
                archive_x=x.copy(), archive_f=f.copy(), archive_g=g.copy(),
            )
        )


def _empty_archive(problem: Benchmark, n_constr: int) -> tuple[Array, Array, Array]:
    return (
        np.empty((0, problem.n_var), dtype=float),
        np.empty((0, problem.n_obj), dtype=float),
        np.empty((0, n_constr), dtype=float),
    )


def _algorithm_rng(seed: int, algorithm: str) -> np.random.Generator:
    tag = int.from_bytes(hashlib.sha256(algorithm.encode("utf-8")).digest()[:4], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, tag]))


def _random_weights(rng: np.random.Generator, n: int, m: int) -> Array:
    if m == 2:
        first = np.linspace(1e-6, 1.0 - 1e-6, n)
        return np.column_stack([first, 1.0 - first])
    weights = rng.dirichlet(np.ones(m), size=max(n * 20, 2000))
    chosen = [int(np.argmax(weights[:, 0]))]
    min_dist = np.linalg.norm(weights - weights[chosen[0]], axis=1)
    for _ in range(1, n):
        idx = int(np.argmax(min_dist))
        chosen.append(idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(weights - weights[idx], axis=1))
    return weights[np.asarray(chosen)]


def _fallback_leader(x: Array, f: Array, g: Array, rng: np.random.Generator) -> Array:
    cv = constraint_violation(g)
    best = np.flatnonzero(cv == cv.min())
    return x[int(rng.choice(best))].copy()


def _leader(archive: tuple[Array, Array, Array], population: tuple[Array, Array, Array], rng: np.random.Generator) -> Array:
    ax, af, _ = archive
    if len(ax):
        return select_archive_leader(ax, af, rng)
    return _fallback_leader(*population, rng)


def _initialize(
    algorithm: str,
    problem: Benchmark,
    initial_x: Array,
    seed: int,
    budget: int,
) -> tuple[np.random.Generator, BudgetedEvaluator, TraceRecorder, Array, Array, Array, tuple[Array, Array, Array]]:
    rng = _algorithm_rng(seed, algorithm)
    evaluator = BudgetedEvaluator(problem, budget)
    x = np.asarray(initial_x, dtype=float).copy()
    if len(x) > budget:
        x = x[:budget]
    f, g = evaluator.evaluate(x)
    archive = update_archive(*_empty_archive(problem, g.shape[1]), x, f, g)
    recorder = TraceRecorder(evaluator)
    recorder.record(archive)
    return rng, evaluator, recorder, x, f, g, archive


def _finish(
    algorithm: str,
    seed: int,
    evaluator: BudgetedEvaluator,
    recorder: TraceRecorder,
    archive: tuple[Array, Array, Array],
    metadata: dict[str, object] | None = None,
) -> OptimizationResult:
    recorder.record(archive, force=True)
    return OptimizationResult(
        algorithm=algorithm, seed=seed, evaluations=evaluator.evaluations,
        feasible_evaluations=evaluator.feasible_evaluations,
        elapsed_seconds=evaluator.elapsed(), archive_x=archive[0].copy(),
        archive_f=archive[1].copy(), archive_g=archive[2].copy(),
        checkpoints=recorder.checkpoints, metadata=metadata or {},
    )


def _polynomial_mutation(x: Array, rng: np.random.Generator, eta: float = 20.0) -> Array:
    y = np.asarray(x, dtype=float).copy()
    mask = rng.random(y.shape) < (1.0 / y.shape[1])
    r = rng.random(y.shape)
    delta = np.where(
        r < 0.5,
        np.power(2.0 * r, 1.0 / (eta + 1.0)) - 1.0,
        1.0 - np.power(2.0 * (1.0 - r), 1.0 / (eta + 1.0)),
    )
    y[mask] += delta[mask]
    return np.clip(y, 0.0, 1.0)


def _sbx(parent_a: Array, parent_b: Array, rng: np.random.Generator, eta: float = 15.0) -> Array:
    a, b = np.asarray(parent_a), np.asarray(parent_b)
    child = a.copy()
    active = rng.random(a.shape) < 0.5
    u = rng.random(a.shape)
    beta = np.where(
        u <= 0.5,
        np.power(2.0 * u, 1.0 / (eta + 1.0)),
        np.power(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta + 1.0)),
    )
    child[active] = (0.5 * ((1.0 + beta) * a + (1.0 - beta) * b))[active]
    no_cross = rng.random(len(a)) >= 0.9
    child[no_cross] = a[no_cross]
    return np.clip(child, 0.0, 1.0)


def optimize_nsga2(problem: Benchmark, initial_x: Array, seed: int, budget: int) -> OptimizationResult:
    algorithm = "nsga2"
    n_constr = int(problem.problem.n_ieq_constr)

    class WrappedProblem(Problem):
        def __init__(self) -> None:
            super().__init__(
                n_var=problem.n_var, n_obj=problem.n_obj, n_ieq_constr=n_constr,
                xl=np.zeros(problem.n_var), xu=np.ones(problem.n_var),
            )
            self.feasible_evaluations = 0

        def _evaluate(self, values, out, *args, **kwargs) -> None:  # type: ignore[override]
            del args, kwargs
            out["F"], evaluated_g = problem.evaluate(values)
            if n_constr:
                out["G"] = evaluated_g
            self.feasible_evaluations += int(np.count_nonzero(constraint_violation(evaluated_g) <= 0.0))

    class ArchiveCallback(Callback):
        def __init__(self) -> None:
            super().__init__()
            self.started = time.perf_counter()
            self.next_boundary = 500
            self.archive = _empty_archive(problem, n_constr)
            self.checkpoints: list[Checkpoint] = []

        def notify(self, pymoo_algorithm) -> None:  # type: ignore[override]
            pop_x = np.asarray(pymoo_algorithm.pop.get("X"), dtype=float)
            pop_f = np.asarray(pymoo_algorithm.pop.get("F"), dtype=float)
            pop_g = (
                np.asarray(pymoo_algorithm.pop.get("G"), dtype=float).reshape(len(pop_x), -1)
                if n_constr else np.empty((len(pop_x), 0), dtype=float)
            )
            self.archive = update_archive(*self.archive, pop_x, pop_f, pop_g)
            evaluations = int(pymoo_algorithm.evaluator.n_eval)
            if evaluations >= self.next_boundary or evaluations >= budget:
                while self.next_boundary <= evaluations:
                    self.next_boundary += 500
                self.checkpoints.append(
                    Checkpoint(evaluations, time.perf_counter() - self.started,
                               self.archive[0].copy(), self.archive[1].copy(), self.archive[2].copy())
                )

    wrapped = WrappedProblem()
    callback = ArchiveCallback()
    started = time.perf_counter()
    result = minimize(
        wrapped,
        NSGA2(
            pop_size=len(initial_x), sampling=np.asarray(initial_x, dtype=float).copy(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=1.0 / problem.n_var, eta=20),
            eliminate_duplicates=True,
        ),
        ("n_eval", budget), seed=seed, callback=callback, verbose=False,
    )
    evaluations = int(result.algorithm.evaluator.n_eval)
    if evaluations != budget:
        raise RuntimeError(f"pymoo NSGA-II used {evaluations} NFE instead of {budget}")
    elapsed = time.perf_counter() - started
    archive = callback.archive
    if not callback.checkpoints or callback.checkpoints[-1].evaluations != evaluations:
        callback.checkpoints.append(Checkpoint(evaluations, elapsed, archive[0].copy(), archive[1].copy(), archive[2].copy()))
    return OptimizationResult(
        algorithm=algorithm, seed=seed, evaluations=evaluations,
        feasible_evaluations=wrapped.feasible_evaluations, elapsed_seconds=elapsed,
        archive_x=archive[0].copy(), archive_f=archive[1].copy(), archive_g=archive[2].copy(),
        checkpoints=callback.checkpoints, metadata={"implementation": "pymoo-0.6.2"},
    )


def optimize_moead_de(problem: Benchmark, initial_x: Array, seed: int, budget: int) -> OptimizationResult:
    algorithm = "moead_de"
    rng, evaluator, recorder, x, f, g, archive = _initialize(algorithm, problem, initial_x, seed, budget)
    n = len(x)
    weights = _random_weights(rng, n, problem.n_obj)
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighbors = np.argsort(distances, axis=1)[:, :20]
    ideal = f.min(axis=0)
    generation_x: list[Array] = []
    generation_f: list[Array] = []
    generation_g: list[Array] = []
    while evaluator.remaining:
        for i in rng.permutation(n):
            if not evaluator.remaining:
                break
            pool = neighbors[i] if rng.random() < 0.9 else np.arange(n)
            replace = pool[pool != i]
            if len(replace) < 3:
                replace = np.arange(n)[np.arange(n) != i]
            r1, r2, r3 = rng.choice(replace, size=3, replace=False)
            mutant = x[r1] + 0.5 * (x[r2] - x[r3])
            mask = rng.random(problem.n_var) < 1.0
            mask[rng.integers(problem.n_var)] = True
            trial = np.where(mask, mutant, x[i])
            trial = _polynomial_mutation(trial[None, :], rng)[0]
            tf, tg = evaluator.evaluate(trial[None, :])
            ideal = np.minimum(ideal, tf[0])
            candidates = pool.copy()
            rng.shuffle(candidates)
            replacements = 0
            for j in candidates:
                if constrained_better(tf[0], tg[0], f[j], g[j], weights[j], ideal):
                    x[j], f[j], g[j] = trial, tf[0], tg[0]
                    replacements += 1
                    if replacements >= 2:
                        break
            generation_x.append(trial.copy())
            generation_f.append(tf[0].copy())
            generation_g.append(tg[0].copy())
        if generation_x:
            archive = update_archive(
                *archive, np.asarray(generation_x), np.asarray(generation_f), np.asarray(generation_g)
            )
            generation_x.clear(); generation_f.clear(); generation_g.clear()
        recorder.record(archive)
    return _finish(algorithm, seed, evaluator, recorder, archive)


def optimize_mopso(problem: Benchmark, initial_x: Array, seed: int, budget: int) -> OptimizationResult:
    algorithm = "mopso"
    rng, evaluator, recorder, x, f, g, archive = _initialize(algorithm, problem, initial_x, seed, budget)
    n = len(x)
    velocity = rng.uniform(-0.1, 0.1, size=x.shape)
    pbest_x, pbest_f, pbest_g = x.copy(), f.copy(), g.copy()
    weights = _random_weights(rng, n, problem.n_obj)
    while evaluator.remaining:
        progress = evaluator.evaluations / budget
        inertia = 0.5 - 0.4 * progress
        count = min(n, evaluator.remaining)
        guides = np.asarray([_leader(archive, (x, f, g), rng) for _ in range(count)])
        velocity[:count] = (
            inertia * velocity[:count]
            + 1.5 * rng.random((count, problem.n_var)) * (pbest_x[:count] - x[:count])
            + 1.5 * rng.random((count, problem.n_var)) * (guides - x[:count])
        )
        velocity[:count] = np.clip(velocity[:count], -0.2, 0.2)
        trial = np.clip(x[:count] + velocity[:count], 0.0, 1.0)
        trial = _polynomial_mutation(trial, rng)
        tf, tg = evaluator.evaluate(trial)
        ideal = np.minimum(f.min(axis=0), tf.min(axis=0))
        for i in range(count):
            if constrained_better(tf[i], tg[i], pbest_f[i], pbest_g[i], weights[i], ideal):
                pbest_x[i], pbest_f[i], pbest_g[i] = trial[i], tf[i], tg[i]
        x[:count], f[:count], g[:count] = trial, tf, tg
        archive = update_archive(*archive, trial, tf, tg)
        recorder.record(archive)
    return _finish(algorithm, seed, evaluator, recorder, archive)


def optimize_mogwo(problem: Benchmark, initial_x: Array, seed: int, budget: int) -> OptimizationResult:
    algorithm = "mogwo"
    rng, evaluator, recorder, x, f, g, archive = _initialize(algorithm, problem, initial_x, seed, budget)
    n = len(x)
    while evaluator.remaining:
        count = min(n, evaluator.remaining)
        leaders = []
        for _ in range(3):
            leaders.append(_leader(archive, (x, f, g), rng))
        a = 2.0 * (1.0 - evaluator.evaluations / budget)
        estimates = []
        for leader in leaders:
            r1, r2 = rng.random((2, count, problem.n_var))
            coefficient_a = 2.0 * a * r1 - a
            coefficient_c = 2.0 * r2
            estimates.append(leader - coefficient_a * np.abs(coefficient_c * leader - x[:count]))
        trial = np.clip(np.mean(estimates, axis=0), 0.0, 1.0)
        tf, tg = evaluator.evaluate(trial)
        x[:count], f[:count], g[:count] = trial, tf, tg
        archive = update_archive(*archive, trial, tf, tg)
        recorder.record(archive)
    return _finish(algorithm, seed, evaluator, recorder, archive)


def optimize_mopio(problem: Benchmark, initial_x: Array, seed: int, budget: int) -> OptimizationResult:
    algorithm = "mopio"
    rng, evaluator, recorder, x, f, g, archive = _initialize(algorithm, problem, initial_x, seed, budget)
    n = len(x)
    velocity = rng.uniform(-0.1, 0.1, size=x.shape)
    while evaluator.remaining:
        count = min(n, evaluator.remaining)
        progress = evaluator.evaluations / budget
        if progress < 0.7:
            guides = np.asarray([_leader(archive, (x, f, g), rng) for _ in range(count)])
            decay = np.exp(-0.2 * (evaluator.evaluations / max(n, 1)))
            velocity[:count] = decay * velocity[:count] + rng.random((count, problem.n_var)) * (guides - x[:count])
            trial = x[:count] + velocity[:count]
        else:
            if len(archive[0]):
                crowd = np.ones(len(archive[0]))
                center = np.average(archive[0], axis=0, weights=crowd)
            else:
                center = _fallback_leader(x, f, g, rng)
            contraction = max(0.05, 1.0 - progress)
            trial = x[:count] + contraction * rng.random((count, 1)) * (center - x[:count])
        trial = _polynomial_mutation(np.clip(trial, 0.0, 1.0), rng)
        tf, tg = evaluator.evaluate(trial)
        x[:count], f[:count], g[:count] = trial, tf, tg
        archive = update_archive(*archive, trial, tf, tg)
        recorder.record(archive)
    return _finish(algorithm, seed, evaluator, recorder, archive)


def optimize_mo_sfmo(problem: Benchmark, initial_x: Array, seed: int, budget: int) -> OptimizationResult:
    """Faithful multi-objective extension retaining repeated grazing/migration."""
    algorithm = "mo_sfmo"
    rng, evaluator, recorder, x, f, g, archive = _initialize(algorithm, problem, initial_x, seed, budget)
    n = len(x)
    weights = _random_weights(rng, n, problem.n_obj)
    compensation = False
    iteration = 0
    estimated_iterations = max(1, (budget - n) // max(n * 21, 1))
    while evaluator.remaining:
        iteration += 1
        progress = min(1.0, iteration / estimated_iterations)
        leader_step = 0.075 - (0.075 - 0.0005) * progress
        follower_step = 0.05 - (0.05 - 0.0005) * progress
        migration_step = 0.05 - (0.05 - 0.0005) * progress
        if compensation:
            leader_step *= 2.0
            follower_step *= 2.0
        start_x, start_f, start_g = x.copy(), f.copy(), g.copy()
        personal_x, personal_f, personal_g = x.copy(), f.copy(), g.copy()
        success = np.zeros(n, dtype=bool)
        leader_position = _leader(archive, (x, f, g), rng)
        leader_index = int(np.argmin(np.linalg.norm(x - leader_position, axis=1)))
        for _ in range(20):
            if not evaluator.remaining:
                break
            count = min(n, evaluator.remaining)
            steps = np.full((count, 1), follower_step)
            if leader_index < count:
                steps[leader_index, 0] = leader_step
            centers = personal_x[:count] if compensation else start_x[:count]
            trial = np.clip(centers + steps * rng.uniform(-1.0, 1.0, size=centers.shape), 0.0, 1.0)
            tf, tg = evaluator.evaluate(trial)
            ideal = np.minimum(personal_f.min(axis=0), tf.min(axis=0))
            for i in range(count):
                if constrained_better(tf[i], tg[i], personal_f[i], personal_g[i], weights[i], ideal):
                    personal_x[i], personal_f[i], personal_g[i] = trial[i], tf[i], tg[i]
                    success[i] = True
            archive = update_archive(*archive, trial, tf, tg)
            recorder.record(archive)
            if np.any(success) and float(np.mean(success)) > 0.3:
                break
        if not evaluator.remaining:
            break
        directions = personal_x - start_x
        lengths = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = np.divide(directions, lengths, out=np.zeros_like(directions), where=lengths > 1e-15)
        if np.any(success):
            # Retain the original repeated collective migration, bounded only by NFE.
            while evaluator.remaining:
                count = min(n, evaluator.remaining)
                guide = _leader(archive, (x, f, g), rng)
                chain = np.empty_like(x[:count])
                previous = guide
                for i in range(count):
                    follow = previous - x[i]
                    norm = np.linalg.norm(follow)
                    if norm > 1e-15:
                        follow /= norm
                    move = 0.5 * directions[i] + 0.5 * follow
                    move_norm = np.linalg.norm(move)
                    if move_norm > 1e-15:
                        move /= move_norm
                    chain[i] = np.clip(x[i] + migration_step * rng.random() * move, 0.0, 1.0)
                    previous = chain[i]
                tf, tg = evaluator.evaluate(chain)
                old_leader_f, old_leader_g = f[leader_index].copy(), g[leader_index].copy()
                x[:count], f[:count], g[:count] = chain, tf, tg
                archive = update_archive(*archive, chain, tf, tg)
                recorder.record(archive)
                if leader_index >= count or not constrained_better(
                    f[leader_index], g[leader_index], old_leader_f, old_leader_g,
                    weights[leader_index], np.minimum(f.min(axis=0), old_leader_f),
                ):
                    break
            compensation = False
        else:
            count = min(max(0, n - 1), evaluator.remaining)
            followers = np.asarray([i for i in range(n) if i != leader_index])[:count]
            if len(followers):
                trial = np.clip(
                    x[followers]
                    + progress * rng.random((len(followers), 1)) * (x[leader_index] - x[followers])
                    + (1.0 - progress) * rng.random((len(followers), 1)) * (personal_x[followers] - x[followers]),
                    0.0, 1.0,
                )
                tf, tg = evaluator.evaluate(trial)
                x[followers], f[followers], g[followers] = trial, tf, tg
                archive = update_archive(*archive, trial, tf, tg)
                recorder.record(archive)
            compensation = True
    return _finish(
        algorithm, seed, evaluator, recorder, archive,
        {"outer_iterations": iteration, "repeated_grazing": True, "repeated_migration": True},
    )


ALGORITHMS: dict[str, Callable[[Benchmark, Array, int, int], OptimizationResult]] = {
    "mo_sfmo": optimize_mo_sfmo,
    "nsga2": optimize_nsga2,
    "moead_de": optimize_moead_de,
    "mopso": optimize_mopso,
    "mogwo": optimize_mogwo,
    "mopio": optimize_mopio,
}


def run_baseline(name: str, problem: Benchmark, initial_x: Array, seed: int, budget: int) -> OptimizationResult:
    try:
        function = ALGORITHMS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown baseline algorithm: {name}") from exc
    return function(problem, initial_x, seed, budget)
