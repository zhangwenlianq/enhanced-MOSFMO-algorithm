"""Locked benchmark definitions and normalized-coordinate evaluation interface."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


Array = NDArray[np.float64]


@dataclass(frozen=True)
class BenchmarkDefinition:
    instance_id: str
    benchmark_id: str
    problem_name: str
    n_var: int
    n_obj: int
    role: str
    difficulty: int | None = None


DEFINITIONS: dict[str, BenchmarkDefinition] = {
    "zdt1_d30": BenchmarkDefinition("zdt1_d30", "zdt", "zdt1", 30, 2, "development"),
    "zdt3_d30": BenchmarkDefinition("zdt3_d30", "zdt", "zdt3", 30, 2, "development"),
    "zdt4_d10": BenchmarkDefinition("zdt4_d10", "zdt", "zdt4", 10, 2, "development"),
    "zdt2_d30": BenchmarkDefinition("zdt2_d30", "zdt", "zdt2", 30, 2, "holdout"),
    "zdt6_d10": BenchmarkDefinition("zdt6_d10", "zdt", "zdt6", 10, 2, "holdout"),
    "zdt1_d100": BenchmarkDefinition("zdt1_d100", "zdt", "zdt1", 100, 2, "robustness"),
    "zdt4_d50": BenchmarkDefinition("zdt4_d50", "zdt", "zdt4", 50, 2, "robustness"),
    "dtlz2_m3_d12": BenchmarkDefinition("dtlz2_m3_d12", "dtlz", "dtlz2", 12, 3, "development"),
    "dtlz7_m3_d22": BenchmarkDefinition("dtlz7_m3_d22", "dtlz", "dtlz7", 22, 3, "development"),
    "dtlz1_m3_d7": BenchmarkDefinition("dtlz1_m3_d7", "dtlz", "dtlz1", 7, 3, "holdout"),
    "dtlz3_m3_d12": BenchmarkDefinition("dtlz3_m3_d12", "dtlz", "dtlz3", 12, 3, "holdout"),
    "dtlz2_m5_d14": BenchmarkDefinition("dtlz2_m5_d14", "dtlz", "dtlz2", 14, 5, "robustness"),
    "dascmop1_d30_i4": BenchmarkDefinition("dascmop1_d30_i4", "dascmop", "dascmop1", 30, 2, "development", 4),
    "dascmop4_d30_i8": BenchmarkDefinition("dascmop4_d30_i8", "dascmop", "dascmop4", 30, 2, "holdout", 8),
    "dascmop7_d30_i12": BenchmarkDefinition("dascmop7_d30_i12", "dascmop", "dascmop7", 30, 3, "holdout", 12),
    "dascmop7_d50_i16": BenchmarkDefinition("dascmop7_d50_i16", "dascmop", "dascmop7", 50, 3, "robustness", 16),
}


class Benchmark:
    """A pymoo problem exposed through normalized [0, 1]^D coordinates."""

    def __init__(self, definition: BenchmarkDefinition) -> None:
        self.definition = definition
        d = definition
        if d.problem_name.startswith("dascmop"):
            self.problem = get_problem(d.problem_name, d.difficulty)
            if d.n_var != self.problem.n_var:
                # The published DAS-CMOP equations are dimension-scalable although
                # pymoo's constructor fixes D=30.  Adjust only the declared shape;
                # its vectorized official equations already use self.n_var.
                self.problem.n_var = d.n_var
                self.problem.xl = np.zeros(d.n_var, dtype=float)
                self.problem.xu = np.ones(d.n_var, dtype=float)
        elif d.problem_name.startswith("dtlz"):
            self.problem = get_problem(d.problem_name, n_var=d.n_var, n_obj=d.n_obj)
        else:
            self.problem = get_problem(d.problem_name, n_var=d.n_var)
        self.lower = np.asarray(self.problem.xl, dtype=float).reshape(-1)
        self.upper = np.asarray(self.problem.xu, dtype=float).reshape(-1)
        if self.lower.size == 1:
            self.lower = np.full(d.n_var, float(self.lower[0]))
        if self.upper.size == 1:
            self.upper = np.full(d.n_var, float(self.upper[0]))
        if self.lower.shape != (d.n_var,) or self.upper.shape != (d.n_var,):
            raise ValueError(f"Invalid bounds for {d.instance_id}")

    @property
    def n_var(self) -> int:
        return self.definition.n_var

    @property
    def n_obj(self) -> int:
        return self.definition.n_obj

    @property
    def benchmark_id(self) -> str:
        return self.definition.benchmark_id

    def denormalize(self, x: Array) -> Array:
        values = np.asarray(x, dtype=float)
        return self.lower + np.clip(values, 0.0, 1.0) * (self.upper - self.lower)

    def evaluate(self, x: Array) -> tuple[Array, Array]:
        values = np.atleast_2d(np.asarray(x, dtype=float))
        if values.shape[1] != self.n_var or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid decision matrix for {self.definition.instance_id}")
        physical = self.denormalize(values)
        result = self.problem.evaluate(physical, return_as_dictionary=True)
        f = np.asarray(result["F"], dtype=float).reshape(values.shape[0], self.n_obj)
        raw_g = result.get("G")
        g = (
            np.empty((values.shape[0], 0), dtype=float)
            if raw_g is None
            else np.asarray(raw_g, dtype=float).reshape(values.shape[0], -1)
        )
        if not np.all(np.isfinite(f)) or not np.all(np.isfinite(g)):
            raise ValueError(f"Non-finite benchmark output for {self.definition.instance_id}")
        return f, g

    @lru_cache(maxsize=1)
    def reference_front(self) -> Array:
        name = self.definition.problem_name
        if name == "zdt3":
            x = np.linspace(0.0, 1.0, 10001)
            candidate = np.column_stack([x, 1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x)])
            nd = NonDominatedSorting().do(candidate, only_non_dominated_front=True)
            pf = candidate[np.asarray(nd, dtype=int)]
        elif name.startswith("zdt"):
            pf = self.problem.pareto_front(n_pareto_points=10001)
        elif name in {"dtlz1", "dtlz2", "dtlz3"}:
            h = 15 if self.n_obj == 5 else 140
            dirs = get_reference_directions("das-dennis", self.n_obj, n_partitions=h)
            pf = self.problem.pareto_front(dirs)
        else:
            pf = self.problem.pareto_front()
        result = np.asarray(pf, dtype=float)
        if result.ndim != 2 or result.shape[1] != self.n_obj or result.size == 0:
            raise ValueError(f"Invalid reference front for {self.definition.instance_id}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Non-finite reference front for {self.definition.instance_id}")
        return result.copy()


def get_benchmark(instance_id: str) -> Benchmark:
    try:
        return Benchmark(DEFINITIONS[instance_id])
    except KeyError as exc:
        raise KeyError(f"Unknown locked benchmark instance: {instance_id}") from exc


def paired_initial_population(instance_id: str, seed: int, size: int = 100) -> Array:
    """Generate the same initial sample for every algorithm in a paired run."""
    definition = DEFINITIONS[instance_id]
    # Separate initialization from algorithm RNG while remaining deterministic.
    sequence = np.random.SeedSequence([seed, 0x53464D4F, definition.n_var])
    return np.random.default_rng(sequence).random((size, definition.n_var))
