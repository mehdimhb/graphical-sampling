# Concentrated-sample translation experiment

Complete: three asymmetric compact samples, three large valid shifts each;12 primary fits plus4 controls. All9 pairs have identical partitions and DI. No genuinely changed reference was found. See REPORT.md, summary.csv and translation_comparison.pdf/png.

Original studies, package, manuscript and reviewer responses remain unchanged. The runner is resumable and skips all completed fit IDs. Inspect saved results without optimization:
```bash
cd /home/bardia/projects/graphical-sampling
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_concentrated_JSSAM_20261016/verify.py
```

All inputs and seeds: config.json. All completed runs: runs/. Protected-file baseline: protected_before.json. Source and result hashes: manifest.json.
