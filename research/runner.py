"""Command-to-structured-result adapter for the locked SFMO research contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pymoo
import scipy

from research.baselines import ALGORITHMS, OptimizationResult, run_baseline
from research.benchmarks import DEFINITIONS, get_benchmark, paired_initial_population
from research.metrics import MetricContext
from research.statistics import friedman_test, holm_adjust, paired_wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / ".autoresearch" / "contract.lock.json"
FINGERPRINT_PATH = PROJECT_ROOT / ".autoresearch" / "contract.sha256"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def contract_fingerprint() -> str:
    return FINGERPRINT_PATH.read_text(encoding="utf-8").strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def algorithm_function(name: str):
    if name in ALGORITHMS:
        return lambda problem, initial_x, seed, budget: run_baseline(name, problem, initial_x, seed, budget)
    if name == "ash_mosfmo":
        module = importlib.import_module("mosfmo.algorithm")
        return module.optimize
    raise KeyError(f"Unknown algorithm: {name}")


def metric_context(instance_id: str) -> MetricContext:
    problem = get_benchmark(instance_id)
    return MetricContext.from_reference_front(problem.reference_front())


def summarize_run(problem, result: OptimizationResult, budget: int) -> tuple[dict[str, float | None], dict[str, float | bool]]:
    context = MetricContext.from_reference_front(problem.reference_front())
    checkpoint_values: list[tuple[int, float, float]] = []
    hit_time: float | None = None
    hit_nfe: float | None = None
    final_igd = float("inf")
    final_hv = 0.0
    for checkpoint in result.checkpoints:
        hv, igd = context.indicators(checkpoint.archive_f, checkpoint.archive_g)
        checkpoint_values.append((checkpoint.evaluations, checkpoint.elapsed_seconds, hv))
        final_hv, final_igd = hv, igd
        if hit_time is None and hv >= 0.95:
            hit_time = checkpoint.elapsed_seconds
            hit_nfe = float(checkpoint.evaluations)
    x_axis = np.asarray([0.0] + [float(item[0]) for item in checkpoint_values])
    y_axis = np.asarray([0.0] + [item[2] for item in checkpoint_values])
    auc = float(np.trapezoid(y_axis, x_axis) / budget) if len(x_axis) > 1 else 0.0
    success = bool(len(result.archive_f) > 0)
    metrics: dict[str, float | None] = {
        "normalized_hv": float(final_hv),
        "igd_plus": float(final_igd) if np.isfinite(final_igd) else None,
        "runtime_seconds": float(result.elapsed_seconds),
        "time_to_95_hv_seconds": None if hit_time is None else float(hit_time),
        "nfe_to_95_hv": hit_nfe,
        "evaluation_count": float(result.evaluations),
        "feasible_solution_rate": float(result.feasible_evaluations / max(result.evaluations, 1)),
        "feasible_archive_success": 1.0 if success else .0,
        "normalized_hv_auc": auc,
    }
    constraints: dict[str, float | bool] = {
        "feasible_solution_rate": metrics["feasible_solution_rate"],
        "feasible_archive_success": success,
        "right_censored_95_hv": hit_time is None,
    }
    return metrics, constraints


def execute_one(algorithm: str, instance_id: str, seed: int, budget: int) -> tuple[OptimizationResult, dict[str, float | None], dict[str, float | bool]]:
    problem = get_benchmark(instance_id)
    initial_x = paired_initial_population(instance_id, seed, 100)
    result = algorithm_function(algorithm)(problem, initial_x, seed, budget)
    if result.evaluations != budget:
        raise RuntimeError(f"{algorithm} used {result.evaluations} evaluations, expected {budget}")
    metrics, constraints = summarize_run(problem, result, budget)
    return result, metrics, constraints


def stage_definition(contract: dict[str, Any], name: str) -> dict[str, Any]:
    for stage in contract["evaluation_stages"]:
        if stage["name"] == name:
            return stage
    raise KeyError(f"Stage not present in contract: {name}")


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        bucket = grouped.setdefault(run["instance_id"], {})
        for key, value in run["metrics"].items():
            if value is not None:
                bucket.setdefault(key, []).append(float(value))
    output: dict[str, Any] = {}
    for instance_id, metrics in grouped.items():
        output[instance_id] = {
            key: {"median": float(np.median(values)), "q1": float(np.quantile(values, 0.25)), "q3": float(np.quantile(values, 0.75))}
            for key, values in metrics.items()
        }
    return output


def run_stage(args: argparse.Namespace) -> int:
    contract = load_contract()
    stage = stage_definition(contract, args.stage)
    started = utc_now()
    candidate_id = args.candidate_id or args.algorithm
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    used = 0
    wall = 0.0
    for instance_id in stage["benchmark_instance_ids"]:
        for seed in stage["seeds"]:
            try:
                result, metrics, constraints = execute_one(args.algorithm, instance_id, seed, int(stage["budget_per_run"]))
                used += result.evaluations
                wall += result.elapsed_seconds
                runs.append({
                    "benchmark_id": DEFINITIONS[instance_id].benchmark_id,
                    "instance_id": instance_id,
                    "seed": seed,
                    "valid": True,
                    "metrics": metrics,
                    "constraints": constraints,
                    "runtime_seconds": result.elapsed_seconds,
                    "evaluation_count": result.evaluations,
                    "resource_usage": {"primary_used": result.evaluations, "details": {"archive_size": len(result.archive_f), **result.metadata}},
                    "error": None,
                })
            except Exception as exc:  # preserve failures in structured output
                errors.append({"instance_id": instance_id, "seed": seed, "type": type(exc).__name__, "message": str(exc)})
                runs.append({
                    "benchmark_id": DEFINITIONS[instance_id].benchmark_id,
                    "instance_id": instance_id,
                    "seed": seed,
                    "valid": False,
                    "metrics": {name: None for name in ["normalized_hv", "igd_plus", "runtime_seconds", "time_to_95_hv_seconds", "nfe_to_95_hv", "evaluation_count", "feasible_solution_rate", "feasible_archive_success", "normalized_hv_auc"]},
                    "constraints": {"feasible_solution_rate": 0.0, "feasible_archive_success": False, "right_censored_95_hv": True},
                    "runtime_seconds": None,
                    "evaluation_count": None,
                    "resource_usage": {"primary_used": 0, "details": {}},
                    "error": errors[-1],
                })
    valid = not errors
    code_path = PROJECT_ROOT / ("mosfmo/algorithm.py" if args.algorithm == "ash_mosfmo" else ("mosfmo/baseline.py" if args.algorithm == "mo_sfmo" else "research/baselines.py"))
    record = {
        "result_version": "1.0",
        "experiment_id": args.experiment_id or f"{candidate_id}-{args.stage}-{started.replace(':', '').replace('-', '')}",
        "candidate_id": candidate_id,
        "parent_candidate_id": args.parent_candidate_id,
        "stage": args.stage,
        "valid": valid,
        "status": "completed" if valid else "failed",
        "contract_fingerprint": contract_fingerprint(),
        "code_reference": f"sha256:{file_sha256(code_path)}:{code_path.relative_to(PROJECT_ROOT).as_posix()}",
        "started_at": started,
        "completed_at": utc_now(),
        "run_configuration": {
            "benchmark_instance_ids": stage["benchmark_instance_ids"],
            "seeds": stage["seeds"],
            "budget_per_run": stage["budget_per_run"],
            "parameters": {"algorithm": args.algorithm, "schema_stage_semantics": stage["promotion_rules"][0]},
        },
        "resource_budget": {
            "primary_unit": "objective_vector_evaluations",
            "limit": len(stage["benchmark_instance_ids"]) * len(stage["seeds"]) * stage["budget_per_run"],
            "used": used,
            "wall_time_seconds": wall,
        },
        "benchmark_runs": runs,
        "aggregates": aggregate_runs(runs),
        "statistical_comparisons": [],
        "errors": errors,
        "decision_metadata": {"decision": None, "rule_results": [], "reason": None, "decided_at": None},
        "extensions": {},
    }
    Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": valid, "output": str(Path(args.output).resolve()), "runs": len(runs), "nfe": used, "errors": len(errors)}))
    return 0 if valid else 2


def validate_infrastructure() -> int:
    summary = {}
    for instance_id in DEFINITIONS:
        problem = get_benchmark(instance_id)
        x = paired_initial_population(instance_id, 101, 4)
        f, g = problem.evaluate(x)
        context = MetricContext.from_reference_front(problem.reference_front())
        summary[instance_id] = {
            "f_shape": list(f.shape), "g_shape": list(g.shape),
            "reference_shape": list(context.reference_front.shape),
            "reference_hv": context.reference_hv,
            "reference_sha256": hashlib.sha256(np.ascontiguousarray(context.reference_front).tobytes()).hexdigest(),
        }
    report = PROJECT_ROOT / ".autoresearch" / "reports" / "reference-fronts.json"
    report.write_text(json.dumps({"validated_at": utc_now(), "instances": summary}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": True, "instances": summary}, indent=2, sort_keys=True))
    return 0


def capture_environment() -> int:
    output = PROJECT_ROOT / ".autoresearch" / "reports" / "environment.json"
    data = {
        "captured_at": utc_now(), "python": sys.version, "platform": platform.platform(),
        "processor": platform.processor(), "numpy": np.__version__, "pymoo": pymoo.__version__,
        "scipy": scipy.__version__, "environment": {key: os.environ.get(key) for key in ["PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
    }
    output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output)}))
    return 0


def validate_baselines() -> int:
    instance_id, seed, budget = "zdt1_d30", 101, 2000
    algorithms = ["mo_sfmo", "nsga2", "moead_de", "mopso", "mogwo", "mopio"]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for name in algorithms:
        pair = []
        for repetition in range(2):
            try:
                result, metrics, constraints = execute_one(name, instance_id, seed, budget)
                pair.append(result)
                records.append({
                    "algorithm": name, "repetition": repetition, "seed": seed,
                    "evaluation_count": result.evaluations,
                    "runtime_seconds": result.elapsed_seconds,
                    "archive_size": len(result.archive_f),
                    "archive_f_sha256": hashlib.sha256(np.ascontiguousarray(result.archive_f).tobytes()).hexdigest(),
                    "metrics": metrics, "constraints": constraints,
                })
            except Exception as exc:
                failures.append({"algorithm": name, "repetition": repetition, "type": type(exc).__name__, "message": str(exc)})
        if len(pair) == 2:
            if not (
                np.array_equal(pair[0].archive_x, pair[1].archive_x)
                and np.array_equal(pair[0].archive_f, pair[1].archive_f)
                and np.array_equal(pair[0].archive_g, pair[1].archive_g)
                and pair[0].evaluations == pair[1].evaluations == budget
            ):
                failures.append({"algorithm": name, "type": "DeterminismFailure", "message": "paired reruns differ or violate budget"})
    output = PROJECT_ROOT / ".autoresearch" / "results" / "smoke" / "baseline-determinism.json"
    payload = {
        "validation_version": "1.0", "completed_at": utc_now(),
        "contract_fingerprint": contract_fingerprint(), "instance_id": instance_id,
        "seed": seed, "budget_per_run": budget, "expected_total_nfe": 24000,
        "actual_total_nfe": int(sum(item["evaluation_count"] for item in records)),
        "valid": not failures, "records": records, "failures": failures,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": not failures, "output": str(output), "runs": len(records), "nfe": payload["actual_total_nfe"], "failures": failures}))
    return 0 if not failures else 2


def capture_initial_baseline() -> int:
    paths = [
        "sfmo.py", "mosfmo/baseline.py", "research/benchmarks.py",
        "research/metrics.py", "research/baselines.py", "research/runner.py",
        "research/statistics.py", "requirements.lock.txt", "research-contract.confirmed.json",
    ]
    records = []
    for relative in paths:
        path = PROJECT_ROOT / relative
        records.append({"path": relative, "size": path.stat().st_size, "sha256": file_sha256(path)})
    payload = {
        "snapshot_version": "1.0", "created_at": utc_now(),
        "contract_fingerprint": contract_fingerprint(), "kind": "checksummed-initial-baseline",
        "files": records,
    }
    output = PROJECT_ROOT / ".autoresearch" / "snapshots" / "initial-baseline-manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output), "files": len(records)}))
    return 0


def analyze_stage(stage: str) -> int:
    directory = PROJECT_ROOT / ".autoresearch" / "results" / stage
    result_files = []
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if value.get("result_version") == "1.0":
            result_files.append((path, value))
    algorithms: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path, value in result_files:
        algorithms.setdefault(value["candidate_id"], {})[path.name] = value["benchmark_runs"]
    metric_names = [
        "normalized_hv", "igd_plus", "runtime_seconds", "time_to_95_hv_seconds",
        "nfe_to_95_hv", "evaluation_count", "feasible_solution_rate",
    ]
    summary: dict[str, Any] = {}
    for candidate_id, records in algorithms.items():
        runs = [run for collection in records.values() for run in collection]
        summary[candidate_id] = {}
        for metric in metric_names:
            values = [run["metrics"].get(metric) for run in runs if run.get("valid")]
            finite = [float(value) for value in values if value is not None and np.isfinite(value)]
            summary[candidate_id][metric] = {
                "count": len(values), "finite_count": len(finite),
                "median": float(np.median(finite)) if finite else None,
                "q1": float(np.quantile(finite, 0.25)) if finite else None,
                "q3": float(np.quantile(finite, 0.75)) if finite else None,
            }
    statistics: dict[str, Any] = {}
    if len(algorithms) >= 3:
        for metric in ["normalized_hv", "igd_plus", "runtime_seconds"]:
            per_algorithm = {}
            for candidate_id, records in algorithms.items():
                by_instance: dict[str, list[float]] = {}
                for run in [item for collection in records.values() for item in collection]:
                    value = run.get("metrics", {}).get(metric)
                    if run.get("valid") and value is not None and np.isfinite(value):
                        by_instance.setdefault(run["instance_id"], []).append(float(value))
                per_algorithm[candidate_id] = [float(np.median(by_instance[key])) for key in sorted(by_instance)]
            lengths = {len(values) for values in per_algorithm.values()}
            statistics[metric] = {"friedman": friedman_test(per_algorithm) if len(lengths) == 1 else None}
            if "ash-mosfmo-final" in per_algorithm:
                comparisons = []
                raw_p = []
                for baseline_id, values in per_algorithm.items():
                    if baseline_id == "ash-mosfmo-final":
                        continue
                    comparison = paired_wilcoxon(per_algorithm["ash-mosfmo-final"], values)
                    raw_p.append(float(comparison["p_value"]) if comparison["p_value"] is not None else 1.0)
                    comparisons.append({"baseline_id": baseline_id, **comparison})
                for comparison, adjusted in zip(comparisons, holm_adjust(raw_p), strict=True):
                    comparison["holm_adjusted_p"] = adjusted
                statistics[metric]["comparisons"] = comparisons
    payload = {
        "analysis_version": "1.0", "generated_at": utc_now(), "stage": stage,
        "contract_fingerprint": contract_fingerprint(), "result_files": [str(path.relative_to(PROJECT_ROOT)) for path, _ in result_files],
        "summary": summary, "statistics": statistics,
    }
    output = PROJECT_ROOT / ".autoresearch" / "reports" / f"analysis-{stage}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output), "results": len(result_files), "algorithms": len(algorithms)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("capture-environment")
    sub.add_parser("validate-baselines")
    sub.add_parser("capture-initial-baseline")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--stage", required=True, choices=["static", "smoke", "quick", "intermediate", "full", "robustness"])
    run = sub.add_parser("run")
    run.add_argument("--algorithm", required=True)
    run.add_argument("--stage", required=True, choices=["static", "smoke", "quick", "intermediate", "full", "robustness"])
    run.add_argument("--output", required=True)
    run.add_argument("--candidate-id")
    run.add_argument("--parent-candidate-id")
    run.add_argument("--experiment-id")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate":
        return validate_infrastructure()
    if args.command == "capture-environment":
        return capture_environment()
    if args.command == "validate-baselines":
        return validate_baselines()
    if args.command == "capture-initial-baseline":
        return capture_initial_baseline()
    if args.command == "analyze":
        return analyze_stage(args.stage)
    if args.command == "run":
        return run_stage(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
