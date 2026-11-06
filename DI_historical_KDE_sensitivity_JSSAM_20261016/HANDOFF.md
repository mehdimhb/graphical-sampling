# Historical-KDE reference sensitivity handoff

Complete and independently verified: 360/360 requested primary evaluations (100%) plus 3 saved baseline controls. Cases only C16(N144,n16), M10(N155,n10), S100(N959,n100); 40 A/B/C outcomes per case. No C32 evaluation, OT optimization, clustering, medoid/matching optimization, TSP search, new sampling or original Monte Carlo rerun occurred.

All saved samples, probabilities, references and translations were reused after exact checks. Original normalization was reproduced from the historical Population pure function. Translated coordinates were not renormalized. The historical radial uniform KDE uses scalar Scott h, historical rtol 1e-4 and equal unit weights. Default rtol 0 gave bitwise-identical densities for all 363 states. DI clipping/sign definitions remain unchanged, without a new zero threshold.

Baselines: C16=0.2766191800930563; M10=-0.27582232823353764; S100=-0.085988696980308352 (full precision in baselines.csv). All 360 signs agree with their own baseline and corresponding old box result; no zero density or nonfinite score occurred. New variability is smaller, but baseline magnitudes change substantially. The revised Appendix needs a new interpretation and the old C32-focused response cannot be retained. These KDE-only results do not reproduce the full historical reference algorithm or validate manuscript Monte Carlo means.

Final outputs: results.csv(360 rows), baselines.csv(3), summary.csv(9), REPORT.md, appendix_table.tex, appendix_discussion_proposed.tex, reviewer_comment1_recommendation.md and appendix_preview.pdf. Proposed text is separate and awaits author approval. No original manuscript or response file was edited. No new study output was pushed or published.

Verification: direct radial and original box counts for all 363 states; extended-precision DI/sign checks; 234 summary-statistic checks and 7,380 pairwise flags; independent source-summary quality and descriptive-statistic review. Every one of 5,387 protected original study/package/manuscript files retains its preflight hash. Resume test skipped all 363 states with zero KDE fits and no changed per-evaluation result files. LaTeX preview compiled twice and was checked by text extraction.

No calculation remains. Inspect/reverify from project root:
```bash
cd /home/bardia/projects/graphical-sampling
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python DI_historical_KDE_sensitivity_JSSAM_20261016/verify.py
```
If recovering an interrupted copy, the density-only resume command is:
```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python DI_historical_KDE_sensitivity_JSSAM_20261016/recalculate.py
```
It validates config/input hashes and completed summaries, skips every completed evaluation and updates only this new output directory. No original runner/pipeline is imported. Do not run the original resume.sh or C32. Derivative summaries/documents can be rebuilt using summarize.py and write_documents.py in this directory; never replace original study files.

The CODE_REVIEW, config, preflight, verification, resume_check, source hash and final preservation records document provenance. The approximately 55% CPU policy was maintained through single-threaded libraries and serial density processing. No manuscript edits are authorized until a new user approval.
