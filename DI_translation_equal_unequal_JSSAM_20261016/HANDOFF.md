# Equal/unequal grid translation experiment: complete

All 14 independent fits and checks are complete: eight primary panels, four tighter unequal-FIP fits, one replay and one joint-coordinate control. The requested figure and report are finalized and visually inspected. All four sample shifts have identical before/after DI and zero changed memberships; no general invariance theorem is claimed. The eight primary references were independently reconstructed.

Main outputs: equal_unequal_translation.pdf/png, REPORT.md, summary.csv, population_FIPs.csv. Full memberships, medoids and matching: display_units.csv and display_references.csv. All 5526 protected files are unchanged. No package, manuscript, original result or previous figure was edited. One thread/worker only.

From the project root, inspect or regenerate without any reference search:
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_equal_unequal_JSSAM_20261016/verify.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_equal_unequal_JSSAM_20261016/make_outputs.py
```
The preserved run_experiment.py skips all completed run IDs; no continuation is needed.
