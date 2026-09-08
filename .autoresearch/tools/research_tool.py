#!/usr/bin/env python3
"""Deterministic utilities for autonomous-algorithm-research.

This module validates and records research artifacts. It never invokes Codex,
runs an experiment, evaluates a candidate, or chooses a research action.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name == "tools" and SCRIPT_DIR.parent.name == ".autoresearch":
    SKILL_ROOT = SCRIPT_DIR.parent
    SCHEMA_DIR = SKILL_ROOT / "schemas"
    TEMPLATE_DIR = SKILL_ROOT / "templates"
else:
    SKILL_ROOT = SCRIPT_DIR.parent
    SCHEMA_DIR = SKILL_ROOT / "assets" / "schemas"
    TEMPLATE_DIR = SKILL_ROOT / "assets" / "templates"
SCHEMAS = {
    "contract": "research-contract.schema.json",
    "state": "state.schema.json",
    "adapter": "adapter.schema.json",
    "candidate": "candidate.schema.json",
    "manifest": "protected-manifest.schema.json",
    "result": "result.schema.json",
    "hypothesis": "hypothesis.schema.json",
}
ACTIVE_PROTECTED_PHASES = {"BASELINE_VALIDATION", "AUTONOMOUS_RESEARCH", "FINALIZATION"}
TERMINAL_PHASES = {"COMPLETED", "BLOCKED", "INVALID_PROTOCOL"}
MANAGED_AGENTS_BEGIN = "<!-- AUTORESEARCH:BEGIN -->"
MANAGED_AGENTS_END = "<!-- AUTORESEARCH:END -->"
HOOK_DESCRIPTION = "AutoResearch locked-protocol guard"
STOP_DESCRIPTION = "AutoResearch autonomous-continuation guard"


class ResearchToolError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ResearchToolError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ResearchToolError(f"invalid JSON in {path}: {exc}") from exc


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def copy_if_missing(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def enumerate_protected(project_root: Path, contract: dict[str, Any]) -> list[Path]:
    project_root = project_root.resolve()
    selected: dict[str, Path] = {}
    for pattern in contract.get("scope", {}).get("protected_paths", []):
        normalized = pattern.replace("\\", "/")
        if Path(normalized).is_absolute() or normalized.startswith("../") or "/../" in f"/{normalized}/":
            raise ResearchToolError(f"protected path must stay project-relative: {pattern}")
        matches = list(project_root.glob(normalized))
        files: list[Path] = []
        for match in matches:
            if match.is_dir():
                files.extend(item for item in match.rglob("*") if item.is_file())
            elif match.is_file():
                files.append(match)
        if not files:
            raise ResearchToolError(f"protected path resolves to no files: {pattern}")
        for item in files:
            resolved = item.resolve()
            if not _inside(project_root, resolved):
                raise ResearchToolError(f"protected artifact escapes project root: {item}")
            relative = resolved.relative_to(project_root).as_posix()
            if relative.startswith(".git/"):
                raise ResearchToolError("do not include .git internals in protected paths")
            selected[relative] = resolved
    return [selected[key] for key in sorted(selected)]


def _manifest_for(project_root: Path, contract: dict[str, Any], contract_hash: str) -> dict[str, Any]:
    artifacts = []
    for path in enumerate_protected(project_root, contract):
        artifacts.append({"path": path.relative_to(project_root.resolve()).as_posix(), "sha256": file_sha256(path), "size": path.stat().st_size})
    manifest = {"manifest_version": "1.0", "contract_fingerprint": contract_hash, "created_at": utc_now(), "artifacts": artifacts}
    require_valid("manifest", manifest)
    return manifest


def verify_protected_manifest(project_root: Path, contract: dict[str, Any], state: dict[str, Any], contract_hash: str) -> dict[str, Any]:
    research_dir = project_root.resolve() / ".autoresearch"
    manifest_path = research_dir / "protected-manifest.lock.json"
    manifest = load_json(manifest_path)
    require_valid("manifest", manifest)
    manifest_hash = fingerprint(manifest)
    recorded = (research_dir / "protected-manifest.sha256").read_text(encoding="utf-8").strip()
    if manifest_hash != recorded:
        raise ResearchToolError("protected manifest fingerprint mismatch")
    if manifest["contract_fingerprint"] != contract_hash:
        raise ResearchToolError("protected manifest references a different Contract")
    if state.get("protected_manifest_fingerprint") != manifest_hash:
        raise ResearchToolError("state references a different protected manifest")
    expected_paths = {path.relative_to(project_root.resolve()).as_posix() for path in enumerate_protected(project_root, contract)}
    recorded_paths = {item["path"] for item in manifest["artifacts"]}
    if expected_paths != recorded_paths:
        raise ResearchToolError(f"protected artifact set changed: missing={sorted(recorded_paths - expected_paths)}, added={sorted(expected_paths - recorded_paths)}")
    for item in manifest["artifacts"]:
        path = (project_root / item["path"]).resolve()
        if not _inside(project_root, path) or not path.is_file():
            raise ResearchToolError(f"protected artifact missing or escapes root: {item['path']}")
        if path.stat().st_size != item["size"] or file_sha256(path) != item["sha256"]:
            raise ResearchToolError(f"protected artifact changed: {item['path']}")
    return {"fingerprint": manifest_hash, "artifact_count": len(manifest["artifacts"])}


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False


def _valid_datetime(value: str) -> bool:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return "T" in value and parsed.tzinfo is not None and parsed.utcoffset() is not None
    except ValueError:
        return False


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        options = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, option) for option in options):
            return [f"{path}: expected type {options}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} is not in {schema['enum']!r}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        additional = schema.get("additionalProperties", True)
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in properties:
                errors.extend(validate_schema(child, properties[key], child_path))
            elif additional is False:
                errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                errors.extend(validate_schema(child, additional, child_path))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: requires at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: allows at most {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            encoded = [canonical_bytes(item) for item in value]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{path}: items must be unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, child in enumerate(value):
                errors.extend(validate_schema(child, item_schema, f"{path}[{index}]"))
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: string is shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: string is longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: string does not match {schema['pattern']!r}")
        if schema.get("format") == "date-time" and not _valid_datetime(value):
            errors.append(f"{path}: expected an ISO 8601 date-time")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: value is less than {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: value is greater than {schema['maximum']}")
    return errors


def _semantic_errors(kind: str, value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if kind == "contract":
        def duplicates(items: list[Any]) -> bool:
            return len(items) != len(set(items))
        metric_names = [item["name"] for item in value.get("metrics", []) if "name" in item]
        baseline_ids = [item["id"] for item in value.get("baselines", []) if "id" in item]
        instance_ids = [instance["id"] for suite in value.get("benchmarks", []) for instance in suite.get("instances", []) if "id" in instance]
        stage_names = [item["name"] for item in value.get("evaluation_stages", []) if "name" in item]
        for label, values in (("metric names", metric_names), ("baseline IDs", baseline_ids), ("benchmark instance IDs", instance_ids), ("stage names", stage_names)):
            if duplicates(values):
                errors.append(f"$: duplicate {label}")
        known_instances = set(instance_ids)
        known_seeds = set(value.get("repeated_runs", {}).get("seeds", []))
        for index, stage in enumerate(value.get("evaluation_stages", [])):
            unknown_instances = set(stage.get("benchmark_instance_ids", [])) - known_instances
            unknown_seeds = set(stage.get("seeds", [])) - known_seeds
            if unknown_instances:
                errors.append(f"$.evaluation_stages[{index}]: unknown instances {sorted(unknown_instances)}")
            if unknown_seeds:
                errors.append(f"$.evaluation_stages[{index}]: seeds not declared in repeated_runs {sorted(unknown_seeds)}")
        budget = value.get("resource_budget", {})
        if budget.get("per_candidate_max", 0) > budget.get("total", 0):
            errors.append("$.resource_budget: per_candidate_max exceeds total")
        stage_costs = []
        for stage in value.get("evaluation_stages", []):
            cost = len(stage.get("benchmark_instance_ids", [])) * len(stage.get("seeds", [])) * stage.get("budget_per_run", 0) * stage.get("candidate_limit", 0)
            stage_costs.append(cost)
        computed_total = budget.get("fixed_allowance", 0) + sum(stage_costs)
        if "total" in budget and not math.isclose(computed_total, budget["total"], rel_tol=1e-12, abs_tol=1e-12):
            errors.append(f"$.resource_budget.total: expected exact maximum {computed_total} from fixed_allowance plus stage costs")
        candidate_stage_costs = [len(stage.get("benchmark_instance_ids", [])) * len(stage.get("seeds", [])) * stage.get("budget_per_run", 0) for stage in value.get("evaluation_stages", [])]
        computed_per_candidate = sum(candidate_stage_costs)
        if "per_candidate_max" in budget and not math.isclose(computed_per_candidate, budget["per_candidate_max"], rel_tol=1e-12, abs_tol=1e-12):
            errors.append(f"$.resource_budget.per_candidate_max: expected {computed_per_candidate} from one execution of every stage")
        modifiable = set(value.get("scope", {}).get("modifiable_paths", []))
        protected = set(value.get("scope", {}).get("protected_paths", []))
        overlap = modifiable & protected
        if overlap:
            errors.append(f"$.scope: exact paths are both modifiable and protected: {sorted(overlap)}")
    elif kind == "state":
        phase = value.get("phase")
        terminal = value.get("terminal")
        if phase in TERMINAL_PHASES and terminal is None:
            errors.append("$.terminal: required for a terminal phase")
        if phase not in TERMINAL_PHASES and terminal is not None:
            errors.append("$.terminal: must be null for a non-terminal phase")
        if phase == "BLOCKED" and terminal is not None and terminal.get("resume_phase") is None:
            errors.append("$.terminal.resume_phase: required when phase is BLOCKED")
        if phase in {"COMPLETED", "INVALID_PROTOCOL"} and terminal is not None and terminal.get("resume_phase") is not None:
            errors.append(f"$.terminal.resume_phase: must be null when phase is {phase}")
        if phase == "COMPLETED" and terminal is not None and not terminal.get("report_path"):
            errors.append("$.terminal.report_path: required when phase is COMPLETED")
    elif kind == "adapter":
        if value.get("status") == "active":
            commands = value.get("commands", {})
            for name in ("baseline", "candidate"):
                if not commands.get(name):
                    errors.append(f"$.commands.{name}: required when adapter is active")
    elif kind == "result":
        if value.get("valid") and value.get("status") != "completed":
            errors.append("$: valid results must have status 'completed'")
        if value.get("status") == "completed" and value.get("completed_at") is None:
            errors.append("$.completed_at: required for completed results")
    elif kind == "candidate":
        terminal_statuses = {"accepted", "rejected", "inconclusive", "invalid"}
        if value.get("status") in terminal_statuses and value.get("decision") is None:
            errors.append("$.decision: required for a candidate with a terminal disposition")
        if value.get("status") not in terminal_statuses and value.get("decision") is not None:
            errors.append("$.decision: must be null before candidate disposition")
    return errors


def validate_value(kind: str, value: Any) -> list[str]:
    if kind not in SCHEMAS:
        raise ResearchToolError(f"unknown schema kind: {kind}")
    schema_path = SCHEMA_DIR / SCHEMAS[kind]
    schema = load_json(schema_path)
    errors = validate_schema(value, schema)
    if isinstance(value, dict):
        errors.extend(_semantic_errors(kind, value))
    return errors


def validate_result_against_contract(result: dict[str, Any], contract: dict[str, Any], contract_hash: str) -> list[str]:
    errors: list[str] = []
    if result.get("contract_fingerprint") != contract_hash:
        errors.append("$.contract_fingerprint: does not match the locked Contract")
    stages = {stage["name"]: stage for stage in contract.get("evaluation_stages", [])}
    stage = stages.get(result.get("stage"))
    if stage is None:
        errors.append("$.stage: stage is not defined by the locked Contract")
        return errors
    config = result.get("run_configuration", {})
    if config.get("benchmark_instance_ids") != stage.get("benchmark_instance_ids"):
        errors.append("$.run_configuration.benchmark_instance_ids: must exactly match the locked stage")
    if config.get("seeds") != stage.get("seeds"):
        errors.append("$.run_configuration.seeds: must exactly match the locked stage")
    if config.get("budget_per_run") != stage.get("budget_per_run"):
        errors.append("$.run_configuration.budget_per_run: must match the locked stage")
    expected_pairs = {(instance, seed) for instance in stage.get("benchmark_instance_ids", []) for seed in stage.get("seeds", [])}
    instance_to_benchmark = {instance["id"]: suite["id"] for suite in contract.get("benchmarks", []) for instance in suite.get("instances", [])}
    if result.get("valid") and result.get("status") == "completed":
        runs = result.get("benchmark_runs", [])
        actual_pairs = [(run.get("instance_id"), run.get("seed")) for run in runs]
        if len(actual_pairs) != len(set(actual_pairs)):
            errors.append("$.benchmark_runs: duplicate instance/seed pairs")
        if set(actual_pairs) != expected_pairs:
            errors.append(f"$.benchmark_runs: expected exact instance/seed coverage {sorted(expected_pairs)}")
        required_metrics = {metric["name"] for metric in contract.get("metrics", []) if metric.get("role") != "diagnostic"}
        for index, run in enumerate(runs):
            instance_id = run.get("instance_id")
            if run.get("benchmark_id") != instance_to_benchmark.get(instance_id):
                errors.append(f"$.benchmark_runs[{index}].benchmark_id: does not match instance definition")
            missing_metrics = required_metrics - set(run.get("metrics", {}))
            if missing_metrics:
                errors.append(f"$.benchmark_runs[{index}].metrics: missing {sorted(missing_metrics)}")
            if not run.get("valid"):
                errors.append(f"$.benchmark_runs[{index}].valid: authoritative completed result contains an invalid run")
            if run.get("runtime_seconds") is None:
                errors.append(f"$.benchmark_runs[{index}].runtime_seconds: required for valid completed runs")
            if run.get("resource_usage", {}).get("primary_used", math.inf) > stage.get("budget_per_run", -1):
                errors.append(f"$.benchmark_runs[{index}].resource_usage.primary_used: exceeds locked per-run budget")
        expected_limit = stage.get("budget_per_run", 0) * len(expected_pairs)
        resource = result.get("resource_budget", {})
        if resource.get("primary_unit") != contract.get("resource_budget", {}).get("primary_unit"):
            errors.append("$.resource_budget.primary_unit: does not match the Contract")
        if resource.get("limit") != expected_limit:
            errors.append(f"$.resource_budget.limit: expected {expected_limit}")
        measured = sum(run.get("resource_usage", {}).get("primary_used", 0) for run in runs)
        if resource.get("used") != measured:
            errors.append(f"$.resource_budget.used: expected sum of run usage {measured}")
    return errors


def require_valid(kind: str, value: Any) -> None:
    errors = validate_value(kind, value)
    if errors:
        raise ResearchToolError(f"{kind} validation failed:\n- " + "\n- ".join(errors))


def _placeholder_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and re.fullmatch(r"<[^<>]+>", value.strip()):
        found.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(_placeholder_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_placeholder_paths(child, f"{path}[{index}]"))
    return found


def _agents_block() -> str:
    return f"""{MANAGED_AGENTS_BEGIN}
## Autonomous algorithm research integrity

When `.autoresearch/contract.lock.json` exists, treat it, its fingerprint, the protected manifest, and recorded evidence as immutable. Run `python .autoresearch/tools/research_tool.py check-integrity --project-root .` before resume, evaluation, acceptance, and finalization; do not substitute a raw file hash. Use the project-local utility for state commits and append-only artifacts instead of editing them directly. Follow the Contract's modification and protected scopes. During active autonomous research, continue from `.autoresearch/state.json` without requesting ordinary scientific decisions from the user. Never weaken benchmarks, metrics, baselines, seeds, budgets, tolerances, acceptance rules, or reporting.
{MANAGED_AGENTS_END}
"""


def install_agents_guidance(project_root: Path, research_dir: Path) -> str:
    target = project_root / "AGENTS.md"
    block = _agents_block()
    if not target.exists():
        atomic_write_text(target, block)
        return "created"
    try:
        existing = target.read_text(encoding="utf-8")
    except UnicodeError:
        atomic_write_text(research_dir / "AGENTS.proposed.md", block)
        return "proposed"
    if MANAGED_AGENTS_BEGIN in existing:
        return "present"
    separator = "" if not existing or existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    atomic_write_text(target, existing + separator + block)
    return "appended"


def _hook_command(script: Path, subcommand: str) -> str:
    return f'"{Path(sys.executable).resolve()}" "{script.resolve()}" {subcommand}'


def _hook_command_windows(script: Path, subcommand: str) -> str:
    return f'py -3 "{script.resolve()}" {subcommand}'


def install_hooks(project_root: Path, research_dir: Path) -> str:
    tools_dir = research_dir / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    guard = tools_dir / "research_tool.py"
    copy_if_missing(Path(__file__).resolve(), guard)
    hooks_path = project_root / ".codex" / "hooks.json"
    pre_command = _hook_command(guard, "hook-pre-tool")
    stop_command = _hook_command(guard, "hook-stop")
    pre_command_windows = _hook_command_windows(guard, "hook-pre-tool")
    stop_command_windows = _hook_command_windows(guard, "hook-stop")
    pre_entry = {
        "description": HOOK_DESCRIPTION,
        "matcher": ".*",
        "hooks": [{"type": "command", "command": pre_command, "commandWindows": pre_command_windows, "timeout": 10, "statusMessage": "Checking locked research scope"}],
    }
    stop_entry = {
        "description": STOP_DESCRIPTION,
        "hooks": [{"type": "command", "command": stop_command, "commandWindows": stop_command_windows, "timeout": 10, "statusMessage": "Checking autonomous research state"}],
    }
    if hooks_path.exists():
        try:
            config = load_json(hooks_path)
            if not isinstance(config, dict) or not isinstance(config.get("hooks", {}), dict):
                raise ResearchToolError("hooks.json must contain an object-valued 'hooks' property")
        except ResearchToolError:
            proposed = {"hooks": {"PreToolUse": [pre_entry], "Stop": [stop_entry]}}
            atomic_write_json(research_dir / "hooks.proposed.json", proposed)
            return "proposed"
    else:
        config = {"hooks": {}}
    hooks = config.setdefault("hooks", {})
    for event, entry, description in (("PreToolUse", pre_entry, HOOK_DESCRIPTION), ("Stop", stop_entry, STOP_DESCRIPTION)):
        values = hooks.setdefault(event, [])
        if not isinstance(values, list):
            proposed = {"hooks": {"PreToolUse": [pre_entry], "Stop": [stop_entry]}}
            atomic_write_json(research_dir / "hooks.proposed.json", proposed)
            return "proposed"
        managed_index = next((index for index, item in enumerate(values) if isinstance(item, dict) and item.get("description") == description), None)
        if managed_index is None:
            values.append(entry)
        elif values[managed_index] != entry:
            values[managed_index] = entry
    atomic_write_json(hooks_path, config)
    return "installed"


def init_project(project_root: Path, contract_path: Path, install_project_hooks: bool) -> dict[str, Any]:
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ResearchToolError(f"project root is not a directory: {project_root}")
    contract = load_json(contract_path.resolve())
    require_valid("contract", contract)
    placeholders = _placeholder_paths(contract)
    if placeholders:
        raise ResearchToolError("contract contains unresolved template placeholders: " + ", ".join(placeholders))
    research_dir = project_root / ".autoresearch"
    for relative in ("candidates", "results/static", "results/smoke", "results/quick", "results/intermediate", "results/full", "results/robustness", "logs", "reports", "snapshots", "tools", "schemas", "templates"):
        (research_dir / relative).mkdir(parents=True, exist_ok=True)
    contract_hash = fingerprint(contract)
    locked_path = research_dir / "contract.lock.json"
    fingerprint_path = research_dir / "contract.sha256"
    if locked_path.exists():
        existing = load_json(locked_path)
        if fingerprint(existing) != contract_hash:
            raise ResearchToolError("a different locked Contract already exists; invalidate/archive it instead of replacing it")
    else:
        atomic_write_json(locked_path, contract)
    if fingerprint_path.exists() and fingerprint_path.read_text(encoding="utf-8").strip() != contract_hash:
        raise ResearchToolError("existing Contract fingerprint does not match")
    atomic_write_text(fingerprint_path, contract_hash + "\n")
    for schema_name in SCHEMAS.values():
        copy_if_missing(SCHEMA_DIR / schema_name, research_dir / "schemas" / schema_name)
    for template_name in ("adapter.template.json", "research-contract.template.json"):
        copy_if_missing(TEMPLATE_DIR / template_name, research_dir / "templates" / template_name)
    local_tool = research_dir / "tools" / "research_tool.py"
    copy_if_missing(Path(__file__).resolve(), local_tool)
    state_path = research_dir / "state.json"
    if not state_path.exists():
        now = utc_now()
        state = {
            "state_version": "1.0", "contract_fingerprint": contract_hash, "protected_manifest_fingerprint": None, "phase": "BUILD", "state_revision": 0,
            "created_at": now, "updated_at": now, "active_experiment": None,
            "initial_baseline": {"candidate_id": None, "code_reference": None, "result_refs": [], "validated": False},
            "incumbent": {"candidate_id": None, "code_reference": None, "result_refs": []},
            "budget_used": {}, "last_completed_action": "Research Contract locked and project scaffold created",
            "next_action": "Build and validate the project adapter and experimental infrastructure", "terminal": None,
        }
        require_valid("state", state)
        atomic_write_json(state_path, state)
    adapter_path = research_dir / "adapter.json"
    if not adapter_path.exists():
        shutil.copy2(TEMPLATE_DIR / "adapter.template.json", adapter_path)
    for ledger in ("history.jsonl", "hypotheses.jsonl", "lessons.jsonl"):
        (research_dir / ledger).touch(exist_ok=True)
    guidance_status = install_agents_guidance(project_root, research_dir)
    hook_status = install_hooks(project_root, research_dir) if install_project_hooks else "not-requested"
    persisted_phase = load_json(state_path)["phase"]
    return {"ok": True, "project_root": str(project_root), "contract_fingerprint": contract_hash, "phase": persisted_phase, "guidance": guidance_status, "hooks": hook_status}


def seal_protected(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    research_dir = project_root / ".autoresearch"
    contract = load_json(research_dir / "contract.lock.json")
    require_valid("contract", contract)
    contract_hash = fingerprint(contract)
    state = load_json(research_dir / "state.json")
    require_valid("state", state)
    if state["phase"] != "BUILD":
        raise ResearchToolError("protected artifacts may be sealed only during BUILD")
    manifest_path = research_dir / "protected-manifest.lock.json"
    fingerprint_path = research_dir / "protected-manifest.sha256"
    if manifest_path.exists():
        verified = verify_protected_manifest(project_root, contract, state, contract_hash)
        return {"ok": True, "already_sealed": True, **verified}
    manifest = _manifest_for(project_root, contract, contract_hash)
    manifest_hash = fingerprint(manifest)
    atomic_write_json(manifest_path, manifest)
    atomic_write_text(fingerprint_path, manifest_hash + "\n")
    proposed = dict(state)
    proposed["protected_manifest_fingerprint"] = manifest_hash
    proposed["state_revision"] += 1
    proposed["updated_at"] = utc_now()
    proposed["last_completed_action"] = "Protected project artifacts sealed"
    proposed["next_action"] = "Activate the adapter and enter baseline validation"
    proposed_path = research_dir / "state.seal-proposed.json"
    atomic_write_json(proposed_path, proposed)
    try:
        committed = commit_state(project_root, proposed_path)
    except Exception:
        raise
    finally:
        try:
            proposed_path.unlink()
        except FileNotFoundError:
            pass
    return {"ok": True, "already_sealed": False, "fingerprint": manifest_hash, "artifact_count": len(manifest["artifacts"]), "state_revision": committed["state_revision"]}


def _require_phase_artifacts(state: dict[str, Any], adapter: dict[str, Any] | None) -> None:
    phase = state["phase"]
    if phase in {"BASELINE_VALIDATION", "AUTONOMOUS_RESEARCH", "FINALIZATION", "COMPLETED"}:
        if adapter is None or adapter.get("status") != "active":
            raise ResearchToolError(f"phase {phase} requires an active project adapter")
        if not state.get("protected_manifest_fingerprint"):
            raise ResearchToolError(f"phase {phase} requires a sealed protected-artifact manifest")
    if phase in {"AUTONOMOUS_RESEARCH", "FINALIZATION", "COMPLETED"}:
        if not state["initial_baseline"].get("validated"):
            raise ResearchToolError(f"phase {phase} requires a validated initial baseline")
        if not state["incumbent"].get("candidate_id") or not state["incumbent"].get("code_reference"):
            raise ResearchToolError(f"phase {phase} requires a recoverable incumbent")
        if not state["initial_baseline"].get("result_refs") or not state["incumbent"].get("result_refs"):
            raise ResearchToolError(f"phase {phase} requires structured result references for baseline and incumbent")


def _verify_state_result_refs(project_root: Path, state: dict[str, Any], contract: dict[str, Any], contract_hash: str) -> None:
    if state["phase"] not in {"AUTONOMOUS_RESEARCH", "FINALIZATION", "COMPLETED"}:
        return
    research_dir = (project_root.resolve() / ".autoresearch").resolve()
    for label in ("initial_baseline", "incumbent"):
        identity = state[label]
        for reference in identity.get("result_refs", []):
            path = (research_dir / reference).resolve()
            if not _inside(research_dir / "results", path) or not path.is_file():
                raise ResearchToolError(f"{label} result reference is missing or outside results: {reference}")
            result = load_json(path)
            require_valid("result", result)
            cross_errors = validate_result_against_contract(result, contract, contract_hash)
            if cross_errors:
                raise ResearchToolError(f"{label} result is incompatible with Contract:\n- " + "\n- ".join(cross_errors))
            if not result.get("valid") or result.get("status") != "completed":
                raise ResearchToolError(f"{label} result is not valid and completed: {reference}")
            if result.get("candidate_id") != identity.get("candidate_id"):
                raise ResearchToolError(f"{label} result candidate ID mismatch: {reference}")


def check_integrity(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    research_dir = project_root / ".autoresearch"
    contract = load_json(research_dir / "contract.lock.json")
    require_valid("contract", contract)
    actual = fingerprint(contract)
    recorded = (research_dir / "contract.sha256").read_text(encoding="utf-8").strip()
    if actual != recorded:
        raise ResearchToolError(f"Contract fingerprint mismatch: recorded {recorded!r}, actual {actual!r}")
    state = load_json(research_dir / "state.json")
    require_valid("state", state)
    if state["contract_fingerprint"] != actual:
        raise ResearchToolError("state references a different Contract fingerprint")
    adapter_path = research_dir / "adapter.json"
    adapter = None
    if adapter_path.exists():
        adapter = load_json(adapter_path)
        require_valid("adapter", adapter)
    _require_phase_artifacts(state, adapter)
    manifest_info = None
    if state.get("protected_manifest_fingerprint") is not None or state["phase"] in {"BASELINE_VALIDATION", "AUTONOMOUS_RESEARCH", "FINALIZATION", "COMPLETED"}:
        manifest_info = verify_protected_manifest(project_root, contract, state, actual)
    _verify_state_result_refs(project_root, state, contract, actual)
    ledger_counts: dict[str, int] = {}
    for ledger in ("history.jsonl", "hypotheses.jsonl", "lessons.jsonl"):
        records: list[dict[str, Any]] = []
        path = research_dir / ledger
        if path.exists():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ResearchToolError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
                records.append(json.loads(line))
        chain_path = path.with_suffix(path.suffix + ".chain")
        chain_entries = [json.loads(line) for line in chain_path.read_text(encoding="utf-8").splitlines() if line.strip()] if chain_path.exists() else []
        if len(chain_entries) != len(records):
            raise ResearchToolError(f"ledger chain length mismatch for {path}")
        previous = "0" * 64
        for index, (record, entry) in enumerate(zip(records, chain_entries)):
            record_hash = fingerprint(record)
            expected_chain = hashlib.sha256(f"{previous}:{record_hash}".encode("ascii")).hexdigest()
            if entry != {"sequence": index, "record_fingerprint": record_hash, "previous_chain_hash": previous, "chain_hash": expected_chain}:
                raise ResearchToolError(f"ledger chain mismatch for {path} at sequence {index}")
            previous = expected_chain
        ledger_counts[ledger] = len(records)
    return {"ok": True, "contract_fingerprint": actual, "protected_manifest": manifest_info, "phase": state["phase"], "state_revision": state["state_revision"], "adapter_status": adapter["status"] if adapter is not None else "missing", "ledger_counts": ledger_counts}


def commit_state(project_root: Path, proposed_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    research_dir = project_root / ".autoresearch"
    integrity = check_integrity(project_root)
    current_path = research_dir / "state.json"
    current = load_json(current_path)
    proposed = load_json(proposed_path.resolve())
    require_valid("state", proposed)
    if proposed["contract_fingerprint"] != integrity["contract_fingerprint"]:
        raise ResearchToolError("proposed state references a different Contract")
    if proposed["created_at"] != current["created_at"]:
        raise ResearchToolError("proposed state must preserve created_at")
    if proposed["state_revision"] != current["state_revision"] + 1:
        raise ResearchToolError("proposed state_revision must increment by exactly one")
    transitions = {
        "BUILD": {"BUILD", "BASELINE_VALIDATION", "BLOCKED", "INVALID_PROTOCOL"},
        "BASELINE_VALIDATION": {"BASELINE_VALIDATION", "AUTONOMOUS_RESEARCH", "BLOCKED", "INVALID_PROTOCOL"},
        "AUTONOMOUS_RESEARCH": {"AUTONOMOUS_RESEARCH", "FINALIZATION", "BLOCKED", "INVALID_PROTOCOL"},
        "FINALIZATION": {"FINALIZATION", "COMPLETED", "BLOCKED", "INVALID_PROTOCOL"},
        "BLOCKED": {"BLOCKED"},
        "INVALID_PROTOCOL": set(),
        "COMPLETED": set(),
    }
    allowed = set(transitions[current["phase"]])
    if current["phase"] == "BLOCKED":
        resume_phase = current.get("terminal", {}).get("resume_phase")
        if resume_phase:
            allowed.add(resume_phase)
    if proposed["phase"] not in allowed:
        raise ResearchToolError(f"invalid state transition: {current['phase']} -> {proposed['phase']}")
    proposed_time = dt.datetime.fromisoformat(proposed["updated_at"].replace("Z", "+00:00"))
    current_time = dt.datetime.fromisoformat(current["updated_at"].replace("Z", "+00:00"))
    if proposed_time < current_time:
        raise ResearchToolError("proposed updated_at precedes current updated_at")
    adapter_path = research_dir / "adapter.json"
    adapter = load_json(adapter_path) if adapter_path.exists() else None
    if adapter is not None:
        require_valid("adapter", adapter)
    _require_phase_artifacts(proposed, adapter)
    contract = load_json(research_dir / "contract.lock.json")
    _verify_state_result_refs(project_root, proposed, contract, integrity["contract_fingerprint"])
    atomic_write_json(current_path, proposed)
    return {"ok": True, "from_phase": current["phase"], "phase": proposed["phase"], "state_revision": proposed["state_revision"], "contract_fingerprint": integrity["contract_fingerprint"]}


def invalidate_protocol(project_root: Path, reason: str, evidence: list[str]) -> dict[str, Any]:
    project_root = project_root.resolve()
    research_dir = project_root / ".autoresearch"
    state_path = research_dir / "state.json"
    state = load_json(state_path)
    require_valid("state", state)
    if state["phase"] == "COMPLETED":
        raise ResearchToolError("a completed campaign cannot be invalidated in place")
    locked_path = research_dir / "contract.lock.json"
    recorded_hash = None
    fingerprint_path = research_dir / "contract.sha256"
    if fingerprint_path.exists():
        recorded_hash = fingerprint_path.read_text(encoding="utf-8").strip()
    raw_hash = file_sha256(locked_path) if locked_path.exists() else None
    canonical_hash = None
    parse_error = None
    try:
        canonical_hash = fingerprint(load_json(locked_path))
    except Exception as exc:
        parse_error = str(exc)
    now = utc_now()
    report = {
        "report_version": "1.0", "created_at": now, "reason": reason, "evidence": evidence,
        "previous_phase": state["phase"], "recorded_contract_fingerprint": recorded_hash,
        "observed_canonical_fingerprint": canonical_hash, "observed_raw_file_sha256": raw_hash,
        "contract_parse_error": parse_error, "resume_requirement": "Confirm a new Research Proposal and create a new Contract; preserve this invalidated protocol and evidence."
    }
    safe_stamp = now.replace(":", "-")
    report_path = research_dir / "reports" / f"invalid-protocol-{safe_stamp}.json"
    if report_path.exists():
        raise ResearchToolError(f"invalidation report already exists: {report_path}")
    atomic_write_json(report_path, report)
    previous_phase = state["phase"]
    state["phase"] = "INVALID_PROTOCOL"
    state["state_revision"] += 1
    state["updated_at"] = now
    state["active_experiment"] = None
    state["last_completed_action"] = "Protocol invalidated and evidence preserved"
    state["next_action"] = "Await explicit confirmation of a new Research Proposal"
    state["terminal"] = {"reason": reason, "target_reached": False, "resume_requirement": report["resume_requirement"], "report_path": report_path.relative_to(research_dir).as_posix(), "resume_phase": None}
    require_valid("state", state)
    atomic_write_json(state_path, state)
    return {"ok": True, "from_phase": previous_phase, "phase": "INVALID_PROTOCOL", "state_revision": state["state_revision"], "report_path": str(report_path)}


def _acquire_lock(path: Path, timeout: float = 10.0) -> int:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ResearchToolError(f"timed out acquiring registry lock: {path}")
            time.sleep(0.05)


def _append_fsynced(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def append_record(kind: str, registry: Path, record_path: Path) -> dict[str, Any]:
    record = load_json(record_path)
    schema_kind = {"hypothesis": "hypothesis"}.get(kind)
    if schema_kind:
        require_valid(schema_kind, record)
    elif kind not in {"history", "lesson"}:
        raise ResearchToolError(f"unknown record kind: {kind}")
    elif not isinstance(record, dict) or not record.get("id") or not record.get("created_at"):
        raise ResearchToolError(f"{kind} records require non-empty 'id' and 'created_at'")
    registry = registry.resolve()
    project_root = _find_project_root(registry)
    if project_root is None:
        raise ResearchToolError("registry must be inside a locked .autoresearch project")
    expected = {
        "hypothesis": project_root / ".autoresearch" / "hypotheses.jsonl",
        "history": project_root / ".autoresearch" / "history.jsonl",
        "lesson": project_root / ".autoresearch" / "lessons.jsonl",
    }[kind].resolve()
    if registry != expected:
        raise ResearchToolError(f"{kind} records must use registry {expected}")
    contract_hash = fingerprint(load_json(project_root / ".autoresearch" / "contract.lock.json"))
    if kind == "hypothesis" and record.get("contract_fingerprint") != contract_hash:
        raise ResearchToolError("hypothesis references a different Contract")
    registry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry.with_suffix(registry.suffix + ".lock")
    descriptor = _acquire_lock(lock_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock_handle:
            lock_handle.write(str(os.getpid()))
            lock_handle.flush()
            os.fsync(lock_handle.fileno())
        id_key = "hypothesis_id" if kind == "hypothesis" else "id"
        existing_records = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines() if line.strip()] if registry.exists() else []
        for existing in existing_records:
            if existing.get(id_key) == record.get(id_key):
                if canonical_bytes(existing) == canonical_bytes(record):
                    return {"ok": True, "already_present": True, "registry": str(registry), "record_fingerprint": fingerprint(record)}
                raise ResearchToolError(f"duplicate {id_key} with different content: {record.get(id_key)}")
        record_hash = fingerprint(record)
        chain_path = registry.with_suffix(registry.suffix + ".chain")
        chain_lines = [json.loads(line) for line in chain_path.read_text(encoding="utf-8").splitlines() if line.strip()] if chain_path.exists() else []
        previous = chain_lines[-1]["chain_hash"] if chain_lines else "0" * 64
        chain_hash = hashlib.sha256(f"{previous}:{record_hash}".encode("ascii")).hexdigest()
        chain_entry = {"sequence": len(existing_records), "record_fingerprint": record_hash, "previous_chain_hash": previous, "chain_hash": chain_hash}
        _append_fsynced(registry, canonical_bytes(record).decode("utf-8") + "\n")
        _append_fsynced(chain_path, canonical_bytes(chain_entry).decode("utf-8") + "\n")
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    return {"ok": True, "already_present": False, "registry": str(registry), "record_fingerprint": fingerprint(record), "chain_hash": chain_hash}


def write_artifact(kind: str, destination: Path, record_path: Path) -> dict[str, Any]:
    if kind not in {"result", "candidate"}:
        raise ResearchToolError(f"unsupported artifact kind: {kind}")
    record = load_json(record_path)
    require_valid(kind, record)
    destination = destination.resolve()
    project_root = _find_project_root(destination)
    if project_root is None:
        raise ResearchToolError("artifact destination must be inside a locked .autoresearch project")
    research_dir = (project_root / ".autoresearch").resolve()
    allowed_root = (research_dir / ("results" if kind == "result" else "candidates")).resolve()
    if not _inside(allowed_root, destination):
        raise ResearchToolError(f"{kind} artifacts must be written below {allowed_root}")
    contract = load_json(research_dir / "contract.lock.json")
    contract_hash = fingerprint(contract)
    if record.get("contract_fingerprint") != contract_hash:
        raise ResearchToolError(f"{kind} references a different Contract")
    if kind == "result":
        cross_errors = validate_result_against_contract(record, contract, contract_hash)
        if cross_errors:
            raise ResearchToolError("result/Contract validation failed:\n- " + "\n- ".join(cross_errors))
    if destination.exists():
        existing = load_json(destination)
        if canonical_bytes(existing) == canonical_bytes(record):
            return {"ok": True, "already_present": True, "path": str(destination), "fingerprint": fingerprint(record)}
        raise ResearchToolError(f"refusing to replace immutable artifact: {destination}")
    atomic_write_json(destination, record)
    return {"ok": True, "already_present": False, "path": str(destination), "fingerprint": fingerprint(record)}


def _hypothesis_tokens(record: dict[str, Any]) -> set[str]:
    fields = [record.get("observed_weakness", ""), record.get("suspected_mechanism", ""), record.get("proposed_modification", ""), record.get("novelty", "")]
    fields.extend(record.get("affected_components", []))
    joined = " ".join(str(item).lower() for item in fields)
    words = {"w:" + token for token in re.findall(r"[\w-]+", joined, flags=re.UNICODE)}
    compact = re.sub(r"\s+", "", joined)
    ngrams = {"g:" + compact[index:index + 3] for index in range(max(0, len(compact) - 2))}
    components = {"c:" + str(item).strip().lower().replace("\\", "/") for item in record.get("affected_components", [])}
    return words | ngrams | components


def similar_hypotheses(registry: Path, record_path: Path, threshold: float) -> dict[str, Any]:
    record = load_json(record_path)
    require_valid("hypothesis", record)
    target = _hypothesis_tokens(record)
    matches: list[dict[str, Any]] = []
    if registry.exists():
        for line_number, line in enumerate(registry.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResearchToolError(f"invalid JSONL at {registry}:{line_number}: {exc}") from exc
            other = _hypothesis_tokens(existing)
            union = target | other
            score = len(target & other) / len(union) if union else 1.0
            if score >= threshold:
                matches.append({"hypothesis_id": existing.get("hypothesis_id"), "similarity": round(score, 6), "status": existing.get("status")})
    matches.sort(key=lambda item: item["similarity"], reverse=True)
    return {"ok": True, "threshold": threshold, "duplicate_risk": bool(matches), "matches": matches}


def _find_project_root(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".autoresearch" / "contract.lock.json").exists():
            return candidate
    return None


def _normalized_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower().replace("\\", "/")


def _protected_match(payload: str, protected_path: str) -> bool:
    normalized = protected_path.lower().replace("\\", "/").lstrip("./")
    base = re.split(r"[*?[]", normalized, maxsplit=1)[0].rstrip("/")
    return bool(base) and base in payload


def hook_pre_tool() -> int:
    try:
        event = json.load(sys.stdin)
        root = _find_project_root(Path(event.get("cwd") or Path.cwd()))
        if root is None:
            return 0
        research_dir = root / ".autoresearch"
        state = load_json(research_dir / "state.json")
        contract = load_json(research_dir / "contract.lock.json")
        tool_name = str(event.get("tool_name", ""))
        tool_input = event.get("tool_input", {})
        payload = _normalized_payload(tool_input)
        lower_name = tool_name.lower()
        utility_write = ".autoresearch/tools/research_tool.py" in payload and any(command in payload for command in (" append-record", " write-artifact", " commit-state", " seal-protected", " invalidate-protocol"))
        if utility_write:
            return 0
        write_like = bool(re.search(r"(write|edit|patch|delete|remove|move|rename|create|update|upload|put)", lower_name))
        if tool_name.lower() == "bash":
            indicators = (">", "set-content", "add-content", "out-file", "remove-item", "move-item", "copy-item", "new-item", " rm ", " mv ", " cp ", " del ", " erase ", " rmdir ", " tee ", " touch ", "truncate", "sed -i", "git checkout", "git restore", "git clean", "git reset")
            padded = " " + payload + " "
            write_like = any(marker in padded for marker in indicators)
        if not write_like:
            return 0
        always_protected = [
            ".autoresearch/contract.lock.json", ".autoresearch/contract.sha256",
            ".autoresearch/protected-manifest.lock.json", ".autoresearch/protected-manifest.sha256",
            ".autoresearch/state.json", ".autoresearch/history.jsonl", ".autoresearch/history.jsonl.chain",
            ".autoresearch/hypotheses.jsonl", ".autoresearch/hypotheses.jsonl.chain",
            ".autoresearch/lessons.jsonl", ".autoresearch/lessons.jsonl.chain",
            ".autoresearch/results/", ".autoresearch/candidates/", ".autoresearch/tools/research_tool.py"
        ]
        protected = always_protected[:]
        if state.get("phase") in ACTIVE_PROTECTED_PHASES:
            protected.extend(contract.get("scope", {}).get("protected_paths", []))
        hit = next((item for item in protected if _protected_match(payload, item)), None)
        if hit:
            output = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": f"AutoResearch protected path: {hit}. Change runtime state/results instead; invalidate the protocol if its scientific meaning is wrong."}}
            print(json.dumps(output, ensure_ascii=False))
    except Exception as exc:
        serialized = locals().get("payload", "")
        if ".autoresearch/contract.lock.json" in serialized or ".autoresearch/contract.sha256" in serialized:
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": f"AutoResearch failed closed for locked Contract files: {exc}"}}, ensure_ascii=False))
        else:
            print(json.dumps({"systemMessage": f"AutoResearch guard could not verify this tool call: {exc}"}, ensure_ascii=False))
    return 0


def hook_stop() -> int:
    try:
        event = json.load(sys.stdin)
        root = _find_project_root(Path(event.get("cwd") or Path.cwd()))
        if root is None or event.get("stop_hook_active"):
            print("{}")
            return 0
        state = load_json(root / ".autoresearch" / "state.json")
        if state.get("phase") == "AUTONOMOUS_RESEARCH" and state.get("terminal") is None:
            reason = "Autonomous research is still active and no valid termination condition has been reached. Resume from the persisted state and continue the next research iteration according to the Master Skill. Next action: " + state.get("next_action", "reconcile state and continue")
            print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))
        else:
            print("{}")
    except Exception as exc:
        print(json.dumps({"continue": True, "systemMessage": f"AutoResearch Stop guard could not read state: {exc}"}, ensure_ascii=False))
    return 0


def _example_contract() -> dict[str, Any]:
    template = load_json(TEMPLATE_DIR / "research-contract.template.json")
    def replace(value: Any) -> Any:
        if isinstance(value, str) and re.fullmatch(r"<[^<>]+>", value):
            return "example"
        if isinstance(value, dict):
            return {key: replace(child) for key, child in value.items()}
        if isinstance(value, list):
            return [replace(child) for child in value]
        return value
    result = replace(template)
    result["confirmed_at"] = utc_now()
    result["contract_id"] = "self-test-contract"
    result["scope"]["modifiable_paths"] = ["algorithm/"]
    result["scope"]["protected_paths"] = ["evaluation/"]
    return result


def self_test() -> dict[str, Any]:
    for schema_name in SCHEMAS.values():
        load_json(SCHEMA_DIR / schema_name)
    contract = _example_contract()
    require_valid("contract", contract)
    with tempfile.TemporaryDirectory(prefix="autoresearch-self-test-") as temporary:
        root = Path(temporary)
        contract_path = root / "confirmed.json"
        atomic_write_json(contract_path, contract)
        initialized = init_project(root, contract_path, True)
        integrity = check_integrity(root)
        proposed_state = load_json(root / ".autoresearch" / "state.json")
        proposed_state["state_revision"] = 1
        proposed_state["updated_at"] = utc_now()
        proposed_state["last_completed_action"] = "Self-test state update"
        proposed_path = root / "proposed-state.json"
        atomic_write_json(proposed_path, proposed_state)
        committed = commit_state(root, proposed_path)
        hook_event = {"cwd": str(root), "tool_name": "apply_patch", "tool_input": {"command": "*** Update File: .autoresearch/contract.lock.json"}}
        old_stdin, old_stdout = sys.stdin, sys.stdout
        import io
        buffer = io.StringIO()
        try:
            sys.stdin = io.StringIO(json.dumps(hook_event))
            sys.stdout = buffer
            hook_pre_tool()
        finally:
            sys.stdin, sys.stdout = old_stdin, old_stdout
        guard_output = json.loads(buffer.getvalue())
        if guard_output.get("hookSpecificOutput", {}).get("permissionDecision") != "deny":
            raise ResearchToolError("PreToolUse self-test did not deny Contract modification")
        copied_guard = root / ".autoresearch" / "tools" / "research_tool.py"
        if not copied_guard.exists():
            raise ResearchToolError("project guard was not copied")
        return {"ok": True, "schemas": sorted(SCHEMAS), "initialized": initialized, "integrity": integrity, "state_commit": committed, "guard_denied_contract_edit": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--kind", choices=sorted(SCHEMAS), required=True)
    validate_parser.add_argument("--file", type=Path, required=True)
    fingerprint_parser = subparsers.add_parser("fingerprint")
    fingerprint_parser.add_argument("--file", type=Path, required=True)
    init_parser = subparsers.add_parser("init-project")
    init_parser.add_argument("--project-root", type=Path, required=True)
    init_parser.add_argument("--contract", type=Path, required=True)
    init_parser.add_argument("--install-hooks", action="store_true")
    check_parser = subparsers.add_parser("check-integrity")
    check_parser.add_argument("--project-root", type=Path, required=True)
    seal_parser = subparsers.add_parser("seal-protected")
    seal_parser.add_argument("--project-root", type=Path, required=True)
    commit_parser = subparsers.add_parser("commit-state")
    commit_parser.add_argument("--project-root", type=Path, required=True)
    commit_parser.add_argument("--state", type=Path, required=True)
    invalidate_parser = subparsers.add_parser("invalidate-protocol")
    invalidate_parser.add_argument("--project-root", type=Path, required=True)
    invalidate_parser.add_argument("--reason", required=True)
    invalidate_parser.add_argument("--evidence", action="append", default=[])
    append_parser = subparsers.add_parser("append-record")
    append_parser.add_argument("--kind", choices=["hypothesis", "history", "lesson"], required=True)
    append_parser.add_argument("--registry", type=Path, required=True)
    append_parser.add_argument("--record", type=Path, required=True)
    artifact_parser = subparsers.add_parser("write-artifact")
    artifact_parser.add_argument("--kind", choices=["result", "candidate"], required=True)
    artifact_parser.add_argument("--destination", type=Path, required=True)
    artifact_parser.add_argument("--record", type=Path, required=True)
    similar_parser = subparsers.add_parser("similar-hypothesis")
    similar_parser.add_argument("--registry", type=Path, required=True)
    similar_parser.add_argument("--record", type=Path, required=True)
    similar_parser.add_argument("--threshold", type=float, default=0.72)
    subparsers.add_parser("hook-pre-tool")
    subparsers.add_parser("hook-stop")
    subparsers.add_parser("self-test")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hook-pre-tool":
        return hook_pre_tool()
    if args.command == "hook-stop":
        return hook_stop()
    try:
        if args.command == "validate":
            value = load_json(args.file.resolve())
            require_valid(args.kind, value)
            output = {"ok": True, "kind": args.kind, "file": str(args.file.resolve()), "fingerprint": fingerprint(value)}
        elif args.command == "fingerprint":
            value = load_json(args.file.resolve())
            output = {"ok": True, "algorithm": "sha256-canonical-json", "fingerprint": fingerprint(value)}
        elif args.command == "init-project":
            output = init_project(args.project_root, args.contract, args.install_hooks)
        elif args.command == "check-integrity":
            output = check_integrity(args.project_root)
        elif args.command == "seal-protected":
            output = seal_protected(args.project_root)
        elif args.command == "commit-state":
            output = commit_state(args.project_root, args.state)
        elif args.command == "invalidate-protocol":
            output = invalidate_protocol(args.project_root, args.reason, args.evidence)
        elif args.command == "append-record":
            output = append_record(args.kind, args.registry, args.record)
        elif args.command == "write-artifact":
            output = write_artifact(args.kind, args.destination, args.record)
        elif args.command == "similar-hypothesis":
            if not 0 <= args.threshold <= 1:
                raise ResearchToolError("threshold must be between 0 and 1")
            output = similar_hypotheses(args.registry, args.record, args.threshold)
        elif args.command == "self-test":
            output = self_test()
        else:
            raise ResearchToolError(f"unhandled command: {args.command}")
        print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ResearchToolError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
