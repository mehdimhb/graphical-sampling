#12x12 sample-translation visual experiment

Complete. Five candidate samples,15 shifts,24 saved full fits including controls.

Figure: translation_comparison.pdf/png. Findings: REPORT.md. All shifts give unchanged partitions and DI; no changed-reference example was found.

No original experiment should be rerun. Inspect saved states with:
```bash
cd /home/bardia/projects/graphical-sampling
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_visual_JSSAM_20261016/verify.py
```

The resumable run_experiment.py skips all completed fits. Existing manuscript/package/old results are unchanged.
