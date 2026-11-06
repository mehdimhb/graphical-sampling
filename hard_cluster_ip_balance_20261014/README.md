# Hard-cluster IP balance, Measure populations

Read `REPORT.md` for findings and probability/convergence qualifications.

- `summary.csv`: 60 settings, with all requested statistics and completed/attempted replicate counts.
- `compact_summary.csv`: one compact comparison across five populations and EP/UP.
- `cluster_measurements.csv`: cluster cardinalities, IP masses and absolute errors.
- `EP_arithmetic_predictions.csv`: separate exact EP cardinality/mass predictions, independent of OT convergence; never substituted for empirical measurements.
- `run_summary.csv`: all 180 prespecified initial samples, including explicit incomplete states.
- `probability_audit.csv`: actual-code vectors versus simple proportional normalization.
- `inclusion_probabilities.csv`, `population_coordinates.csv`: the exact supplied probabilities and original coordinates.
- `mass_distributions.png/pdf`, `imbalance_vs_fraction.png/pdf`: requested figures.
- `runs/*/state.npz`: completed OT coupling, centroids, sample, path, ordering, fractional and hard memberships.
- `sources/`: package/adapters/helper snapshots and manuscript/probability construction evidence. The manuscript is a read-only provenance copy.
- `cached_OT/`: six preserved prior converged OT states used in this study.
- `verification.csv/json`: independent exact overlap, balance, bound, convergence and source preservation audit.
- `resource_monitor.csv` and `resource_monitor_v2.csv`: two-second total-machine CPU/memory readings and throttling actions.

## Reproduce

The existing project `.venv` was used for all runs. For another environment, use Python 3.12 and the pinned requirements (including the system R runtime needed by the preserved package imports). The source snapshot includes local package version metadata; no editable repository installation is needed.

```bash
python -m pip install -r requirements.txt
python -m pytest -q test_metrics.py
python experiment.py --phase pilot --jobs 1
python experiment.py --phase first --jobs 8
python experiment.py --phase replicates --jobs 8
python analyze.py
python ep_arithmetic.py
python verify.py
python repeat_check.py
python write_report.py
python package_outputs.py
```

The delivered design and six caches fix all initial samples. Completed runs are skipped automatically. To reproduce into fresh outputs, copy this directory, remove only that copy's `runs/` directory and `resource_monitor*.csv`, then execute the commands above. Outside the original workspace use `python verify.py --allow-unavailable-originals`: numerical checks still run, while unavailable originals are explicitly recorded and preservation is not claimed. Retain `design.json`, populations, sources and cached_OT. For n>20, failed first replicates leave later replicates explicitly unattempted; small-n settings still attempt all three samples; no unconverged states are treated as measurements. One can run an individual prespecified case with `python experiment.py --phase case --case-id Swiss_n20_UP_r1` in the fresh copy. Use this individual-case command only in the fresh copy to preserve delivered results. That command is sequential; use the monitored scheduler for batches.

Each case has a 240-second **process CPU** OT budget, excluding pauses and competing applications, and the prior 2000-outer/1,000,000-inner update limits. A budget failure does not imply IPnM cannot converge. `--jobs` is an upper bound, limited to approximately 55% of available logical CPUs; workers each use one computational thread and lower priority. Total measured CPU controls two-second pauses at 58%, a 30-second rolling limit at 57.5%, and adaptive concurrency with persistent suspension of excess workers; memory pressure also pauses workers. Transient readings above 60% can occur from other applications before a pause; the reported sustained values use the recorded 30-second window.

Primary UP exactly follows the existing Measure code: stored `prob`, Meuse Cu, Swiss `clip(AREA,5,100)`, passed to the original installed certainty adjustment helper. The simple proportional vectors that exceed one are separately flagged and are never fed to OT. Swiss raw Area has zero weights; this study did not invent clipping or delete those sites. The existing preprocessing differs from the manuscript's abbreviated statement of Area proportionality, so the report recommends making it explicit.

Summary statistics pool clusters across the available replicates within each setting. Since each replicate has n clusters, this is equal weighting of replicate means. Min/max and maximum error are worst observed across those replicates. SD uses ddof=0 over the measured clusters. Strict threshold counts use a 1e-12 numerical guard solely at the 0.10/0.25 endpoints; masses and errors are never clipped or normalized. Bound verification allows 1e-10 for floating arithmetic; exact assignment is independently verified with rational interval endpoints. A fixed TSP seed and 15 documented approximate restarts are used throughout; no TSP sensitivity or DI calculation is performed.
