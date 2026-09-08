<!-- AUTORESEARCH:BEGIN -->
## Autonomous algorithm research integrity

When `.autoresearch/contract.lock.json` exists, treat it, its fingerprint, the protected manifest, and recorded evidence as immutable. Run `python .autoresearch/tools/research_tool.py check-integrity --project-root .` before resume, evaluation, acceptance, and finalization; do not substitute a raw file hash. Use the project-local utility for state commits and append-only artifacts instead of editing them directly. Follow the Contract's modification and protected scopes. During active autonomous research, continue from `.autoresearch/state.json` without requesting ordinary scientific decisions from the user. Never weaken benchmarks, metrics, baselines, seeds, budgets, tolerances, acceptance rules, or reporting.
<!-- AUTORESEARCH:END -->
