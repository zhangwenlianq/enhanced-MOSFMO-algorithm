# Baseline validation report

- Contract: `sfmo-moo-efficiency-2026-09-08-v1`
- Contract fingerprint: `3e605a2918f27fe21754b5a1a7a94e6e62b19bc7c592d70684d28f0fc4499135`
- Protected-manifest fingerprint: `1d1b35893e1bf4f9cb71feada8d6bc6c99fb07c0b05fb37eb16f4c4005d6193a`
- Initial incumbent: `baseline-mo-sfmo`
- Validation resource use: 24,000 objective-vector evaluations (the complete fixed allowance)

## Gate results

1. **Build/import and tests — PASS.** CPython 3.13 imports the original SFMO, pymoo 0.6.2, NumPy 2.5.2, SciPy 1.18.1, all research modules, and six baselines. Six unit tests cover locked benchmark outputs, reference-front indicators, constraint violation, archive correctness, environmental selection, strict NFE, and baseline determinism.
2. **Required baselines run — PASS.** `mo_sfmo`, `nsga2`, `moead_de`, `mopso`, `mogwo`, and `mopio` each completed twice on `zdt1_d30`, seed 101, budget 2,000.
3. **Benchmark/evaluator semantics — PASS.** All 16 locked instances return finite values of the declared objective and constraint shapes. Frozen reference sets have positive finite reference hypervolume; their array hashes are recorded in `reference-fronts.json`. Internal decisions are normalized to `[0,1]^D`, and all objectives are evaluated in the original physical domain.
4. **Metric/statistic semantics — PASS.** A reference front scores normalized HV 1 and IGD+ 0 on every instance. Empty-feasible and censored behavior is explicit. Friedman, paired Wilcoxon, Holm, and hierarchical-bootstrap utilities import and pass syntax/runtime checks.
5. **Reproducibility — PASS.** All six paired baseline reruns produced byte-identical decision, objective, and constraint archives at the same seed and exact budget. Runtime is deliberately excluded from exact equality.
6. **Resource accounting — PASS.** All 12 smoke runs used exactly 2,000 NFE; the structured record totals exactly 24,000. The runner rejects any unexplained NFE mismatch. Search runtime uses `perf_counter` and fixed numerical thread variables.
7. **Structured durability — PASS.** The static result passed the Contract-bound result schema and was written create-once through `research_tool.py`. Smoke validation, environment capture, reference hashes, logs, and reports are persisted under `.autoresearch`.
8. **Candidate isolation — PASS.** The original `sfmo.py`, multi-objective baseline, benchmark, metrics, baseline comparators, runner, statistics, requirements, and tests are sealed. Candidates are restricted to `mosfmo/algorithm.py` and `mosfmo/candidates/**`.
9. **Protected integrity — PASS.** Ten protected artifacts are sealed by the protected manifest. `check-integrity` validates both canonical Contract fingerprint and protected fingerprint.
10. **Rollback — PASS.** The initial incumbent is a protected entry point backed by `research/baselines.py`; its exact file hashes are stored in `snapshots/initial-baseline-manifest.json`. A rejected candidate never overwrites this baseline and can be discarded by selecting the recorded incumbent reference.

## Diagnostic observations

The original scalar SFMO smoke test used 648 NFE for a nominal 48 grazing trials because collective migration repeats until leader failure. In the locked 2,000-NFE multi-objective smoke, the initial MO-SFMO remained at normalized HV 0 on ZDT1, whereas the protected official pymoo NSGA-II wrapper reached approximately 0.281. This is diagnostic only, not authoritative performance evidence, but it supports the first hypothesis: one-shot behavior is necessary for efficiency and may also improve useful progress per NFE.

## Gate decision

All ten baseline-validation requirements pass. The project may enter `AUTONOMOUS_RESEARCH` with `baseline-mo-sfmo` as the validated initial incumbent.
