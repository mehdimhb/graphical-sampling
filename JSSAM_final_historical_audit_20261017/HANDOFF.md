# Final historical audit handoff

The requested audit is complete. No further simulation or manuscript integration is authorized by this deliverable.

Start with [REPORT.md](REPORT.md), organized A–E as requested, then [PROPOSED_CORRECTIONS.md](PROPOSED_CORRECTIONS.md). The `.tex` fragments are mutually qualified approval-only alternatives, not a combined patch. Exact quotations use the designated 12-October snapshot; the actual latest `RevThree`/`RevFive` source and exact current Comment 5 were not found. Preserve their real wrappers when authoritative source becomes available.

The 162-setting partition is 104 exact-linked MC mean/SD pairs with source-inferred 2000-score counts, two discrepant MC means, 32 unlogged MC settings, and 24 weighted finite designs. Effective MC counts were not independently logged. Older 5000-labelled summaries supply the auxiliary indices. Traced MC uses weighted preliminary clustering, ordinary Euclidean medoids/matching and spherical Scott KDE. Runtime attribution is qualified; finite-design import identity is unresolved.

The deterministic two-pair example proves ordinary Euclidean optimal assignment can change under sample-only translation with fixed medoids. Squared full-bijection assignment preserves its minimizing set, subject to tie handling. Conditional DI invariance requires a fixed partition/reference/pairing and bandwidth; the earlier kernel comparison and numerical translation examples do not establish a universal historical theorem.

Useful artifacts:

- `evidence/replications/per_setting_replication_ledger.csv`: requested/recorded/inferred/unknown count distinction per setting.
- `evidence/matching/row_provenance.csv`, `matching_history.csv`: source/output attribution and historical metric.
- `evidence/kernel/kernel_history.csv`: source-only KDE trace at six snapshots.
- `assignment_counterexample.py`, `.json`, `assignment_costs.csv`: minimal diagnostic, no DI rescoring.
- `proposed_passages.json`, `prepare_corrections.py`: exact source quotations and reproducible proposed bodies.
- `verification.json`, `preservation.json`, `REVIEW.md`: checks, unchanged originals and independent reviews.

From the repository root, reproduce the minimal diagnostic and final checks:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python JSSAM_final_historical_audit_20261017/assignment_counterexample.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python JSSAM_final_historical_audit_20261017/prepare_corrections.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python JSSAM_final_historical_audit_20261017/verify_audit.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python JSSAM_final_historical_audit_20261017/preservation.py
```

Source-extraction reproduction commands are documented in each evidence directory's findings. Those scripts parse Git/JSON/CSV/static numeric buffers without executing notebook code, importing the scientific scorer or unpickling designs. Audit scripts write only this directory. Do not rerun `preservation.py before`: the original baseline is intentionally immutable.

No Monte Carlo samples or DI scores were generated. Existing package, manuscripts, bibliography, responses, datasets, historical results, completed kernel experiment and unrelated workspace changes were preserved. No commit, push or branch change was made. Eight hashing workers use half of the 16 available CPUs for the preservation check; the tiny assignment diagnostic uses one thread to avoid pointless parallel overhead.
