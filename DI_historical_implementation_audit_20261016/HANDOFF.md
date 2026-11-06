# Historical DI audit handoff

Complete. Report: DI_implementation_audit.md. All original manuscripts, package, figures, results, Git history and the 480-run study remain unchanged; 5,287 protected files passed the before/after hash check. No sampling, OT/TSP search or DI evaluation was run.

The report answers A–E explicitly. Strong scoped weighted provenance comes from June-1 historical notebook outputs: 104/106 comparable Monte Carlo cells exactly match current legacy summaries; two means differ by 0.000009 and 0.0005. Another 32 MC cells lack retained comparable outputs, while 24 NMS/GMS rows have a separate uncertain import path. The 162 plotting-input DI rows match the current All_DI table exactly. All 96 synthetic DI bars match the historical medoid master, within PDF rendering precision, in the June-1 plot, later FINAL archive and supplied R1 proof, independently calibrated from axis ticks. Earlier weighted Measure wording was historically consistent; later OT wording creates the scoped method/results discrepancy.

Historical target-n inference can yield n−1, confirmed by arithmetic-only reconstruction in 18/36 settings with today's helper/NumPy2.4.4. This does not establish original incidence or effects on rankings; no per-run reference counts were located. The retained MC output says 2,000 DI replications, so a uniform 5,000 claim is unsupported. Do not use the 480 fixed-sample study as proof of legacy–OT equivalence.

No authorized audit task remains. Author decisions concern accurate method/results disclosure and recovering missing historical records, not an automatic full simulation rerun. No manuscript correction was applied.

Recheck from /home/bardia/projects/graphical-sampling:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python DI_historical_implementation_audit_20261016/audit_checks.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python DI_historical_implementation_audit_20261016/figure_checks.py
.venv/bin/python DI_historical_implementation_audit_20261016/verify_preservation.py
```

These commands read saved evidence and reconstruct size arithmetic only. PDF/SVG regeneration, if needed, is documented in prepare_figure_evidence.sh; it writes exclusively here. The scripts never execute simulation notebook cells. Keep all computation single-threaded and preserve the approximately55% CPU restriction.
