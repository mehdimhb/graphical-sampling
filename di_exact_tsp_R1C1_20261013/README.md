# Exact open-TSP check — Reviewer1 Comment1

Read `01_Report.md` and `02_Comparison.csv`. The same two fixed samples and all scoring settings from the previous experiment are retained. Only the approximate TSP ordering is replaced. No manuscript text or reviewer response is drafted here.

- `exact_tsp.py`: exhaustive, reversal-aware open-TSP solver with80-digit Euclidean lengths and deterministic spatial-lexicographic ties.
- `runner.py`: injects the exact solver into the separately instantiated reference object; calls the same manuscript-specific adapter and actual package routines. This changes an experiment instance, not the package source.
- `run_experiment.py`:20 full repetitions across two samples and RNG seeds0–9, followed by every additional oriented optimal path (two reversal checks here).
- `analyze.py`: compares only the previous20 official approximate runs; stress initializations are excluded.
- `write_report.py`: generates the short report from saved numerical results.
- `verify.py`: independent Held–Karp DP, full long-double enumeration,120-digit optimum checks and per-run numerical audits.
- `frozen_design.json`: exact samples, unchanged settings, tie policy, and original source hashes fixed before new scoring.
- `exact_results.csv`:20 deterministic exact-path repetitions; `optimal_path_checks.csv`:two reversed-optimum full constructions.
- `repeatability.csv`:bit-identical intermediate-state checks; `optimal_orientation_comparisons.csv`:hard/soft reference and score effects of reversing the path.
- `approximate_results.csv`, `approximate_vs_exact.csv`:previous official results and each baseline's objective/reference/membership/DI comparison with exact ordering.
- `enumeration/`:all20160 canonical unoriented path costs per sample and optimal-path/tie metadata. Each canonical path covers two orientations.
- `exact_runs/`, `optimal_path_checks/`:lossless NPZ state, full population coordinates and density contributions, medoid/sample matching and hard IP totals, final fractional memberships, preliminary OT transport, and exact ordering metadata.
- `approximate_baseline/`:unaltered copies of the20 prior official reference states used for comparison.
- `sources/`:verbatim population/manuscript snapshots, unchanged prior adapter modules, settings and source records. The complete old result snapshot includes stress rows for provenance, but scripts explicitly filter them out; no stress case is used or recomputed.
- `verification.json` / `.log`:independent numerical checks and source preservation evidence.

CSV exports use17 significant digits; read with `pd.read_csv(path,float_precision='round_trip')`. Population IDs are1-based source row numbers; array indices, spatial ranks and cluster labels are0-based. Original and normalized coordinates and all transformations are retained in the source settings and per-run data. Box H remains fixed across every comparison.

The previous manuscript/package audit still applies: sample-initialized OT, squared medoids/matching and the uniform box require the isolated adapter rather than unmodified public DI defaults. Every new calculation uses observed sample centers. `sources/adapter.py`, `ot_adapter.py` and `density_helper.py` are byte-identical copies of the previous experiment's code; runtime replacement of the ordering object is the only procedural difference.

To reproduce without overwriting saved results, copy the code, `frozen_design.json` and `sources/` into a fresh directory, then use the existing repository environment. The source paths in the frozen records point to this user's unchanged workspace; on another computer remap those paths in a new copy while preserving this original record.

```bash
mkdir /tmp/DI_EXACT_TSP_fresh
cp *.py frozen_design.json /tmp/DI_EXACT_TSP_fresh/
cp -r sources /tmp/DI_EXACT_TSP_fresh/
export PYTHONPATH=/home/bardia/projects/graphical-sampling
export MPLCONFIGDIR=/tmp/mpl-exact-tsp
/home/bardia/projects/graphical-sampling/.venv/bin/python /tmp/DI_EXACT_TSP_fresh/run_experiment.py
/home/bardia/projects/graphical-sampling/.venv/bin/python /tmp/DI_EXACT_TSP_fresh/analyze.py
/home/bardia/projects/graphical-sampling/.venv/bin/python /tmp/DI_EXACT_TSP_fresh/verify.py
/home/bardia/projects/graphical-sampling/.venv/bin/python /tmp/DI_EXACT_TSP_fresh/write_report.py
```

No figure is needed: the compact six-row table identifies both prior path classes and the deterministic result for each sample. Exact TSP is a focused ordering variant here; no existing package or manuscript was modified.
