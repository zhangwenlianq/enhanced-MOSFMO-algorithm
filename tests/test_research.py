from __future__ import annotations

import unittest

import numpy as np

from research.baselines import ALGORITHMS, run_baseline
from research.benchmarks import DEFINITIONS, get_benchmark, paired_initial_population
from research.metrics import MetricContext, constraint_violation, environmental_select, update_archive


class BenchmarkTests(unittest.TestCase):
    def test_all_locked_instances_are_finite(self) -> None:
        for instance_id, definition in DEFINITIONS.items():
            with self.subTest(instance=instance_id):
                problem = get_benchmark(instance_id)
                x = paired_initial_population(instance_id, 101, 3)
                f, g = problem.evaluate(x)
                self.assertEqual(f.shape, (3, definition.n_obj))
                self.assertEqual(g.shape[0], 3)
                self.assertTrue(np.all(np.isfinite(f)))
                self.assertTrue(np.all(np.isfinite(g)))

    def test_reference_front_metrics(self) -> None:
        for instance_id in DEFINITIONS:
            with self.subTest(instance=instance_id):
                problem = get_benchmark(instance_id)
                context = MetricContext.from_reference_front(problem.reference_front())
                reference = problem.reference_front()
                hv, igd = context.indicators(reference, np.empty((len(reference), 0)))
                self.assertAlmostEqual(hv, 1.0, places=10)
                self.assertAlmostEqual(igd, 0.0, places=12)


class SelectionTests(unittest.TestCase):
    def test_constraint_violation(self) -> None:
        g = np.array([[-1.0, 0.0], [1.0, -2.0], [1.0, 2.0]])
        np.testing.assert_allclose(constraint_violation(g), [0.0, 1.0, 3.0])

    def test_archive_keeps_only_feasible_nondominated(self) -> None:
        x = np.array([[0.0], [0.5], [1.0], [0.2]])
        f = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.8, 0.8]])
        g = np.array([[-1.0], [-1.0], [-1.0], [1.0]])
        empty = (np.empty((0, 1)), np.empty((0, 2)), np.empty((0, 1)))
        ax, af, ag = update_archive(*empty, x, f, g, capacity=10)
        self.assertEqual(len(ax), 3)
        self.assertTrue(np.all(constraint_violation(ag) == 0.0))

    def test_environmental_selection_size(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.random((20, 3)); f = rng.random((20, 2)); g = np.empty((20, 0))
        sx, sf, sg = environmental_select(x, f, g, 10)
        self.assertEqual(sx.shape, (10, 3)); self.assertEqual(sf.shape, (10, 2)); self.assertEqual(sg.shape, (10, 0))


class BaselineTests(unittest.TestCase):
    def test_every_baseline_honors_budget_and_is_deterministic(self) -> None:
        problem = get_benchmark("zdt1_d30")
        initial = paired_initial_population("zdt1_d30", 101, 100)
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                first = run_baseline(name, problem, initial, 101, 300)
                second = run_baseline(name, problem, initial, 101, 300)
                self.assertEqual(first.evaluations, 300)
                self.assertEqual(second.evaluations, 300)
                np.testing.assert_array_equal(first.archive_f, second.archive_f)


if __name__ == "__main__":
    unittest.main()
