# Reviewer 1 Comment 1 — alternative reference sensitivity

Start with [report.md](report.md). This folder addresses only reference choice for the same observed sample. It does not revise the manuscript, the DI formula, the package, the Sampling paper, or any previous result.

The focused experiment fixes two samples before scoring, one clustered N=144 population, n=8 and one UP vector. Ten official-procedure seeds and ten alternative-initialization stress tests are evaluated per sample. All40 runs converged. Candidate suitability is evaluated using clustering losses, not DI. Full preselection and source hashes are in `preselection.json`; numerical settings, bandwidth matrices and coordinate transformations are in `settings.json`.

- `report.md`: concise findings, one compact table, objective-tolerance qualification and theoretical limits.
- `comparison_table.csv` / `.tex`: six selected reference/sample cases; CSV retains17 significant digits and all medoid IDs/hard IP totals. The LaTeX table is proposed green content and has been compiled for syntax verification.
- `reference_comparison.pdf` / `.png`: one three-panel figure with the fixed S2 sample, alternative computed medoids and actual unit density contributions. `figure_panels.csv` records exact axis/color settings and source tables.
- `manuscript_clarification.tex`: separate minimal proposed insertion, wrapped in `\textcolor{green}{...}`; original revisions untouched.
- `reviewer_response.tex`: separate proposed response to this comment only; no claim that the proposed edit is already applied.
- `discrepancies.md`: manuscript/public-package audit and precise operational suitability definitions.
- `theory.md`: relabeling invariance, a sufficient box-neighborhood margin condition, and what objective similarity does not prove.
- `results.csv`: all40 actual runs; `scored_references.csv` adds common-scale loss for comparison only.
- `distinct_references.csv`: one representative per canonical hard partition/medoid reference for each observed sample; mere label permutations excluded.
- `equivalent_runs.csv`: mapping all repeat runs to canonical references and observed numerical ranges.
- `reference_pairs.csv`: all30 distinct within-sample pairs, objective gaps, medoid geometry, partition disagreement, density-count differences and unchanged-formula score decomposition.
- `objective_tolerances.csv`: numerical equality,0.1%,1%,5%, with native-OT-only, joint native-OT/soft-compactness, and common-scale checks; includes eligible counts so an empty set cannot be mistaken for robustness.
- `run_group_summary.csv`: official versus stress results, never pooled as the same initialization policy.
- `population.csv` and `sources/`: actual raw/normalized coordinates, IPs, verbatim raw population, and manuscript snapshot.
- `runs/<run_id>/state.npz`: lossless arrays of preliminary OT transport/centroids/labels, fractional refinement, hard labels, medoids, sample matching, translation vectors and locations, original/translated densities, per-unit contributions, H, TSP path and unit order.
- `runs/<run_id>/population_density.csv`: every original/translated raw and normalized unit coordinate, IDs, IPs, hard membership and density contributions.
- `runs/<run_id>/memberships.csv`: all preliminary transport and final fractional shares, including zeros.
- `runs/<run_id>/reference_matching.csv`: medoid/sample IDs and raw coordinates, translation vectors, soft/hard IP totals, squared medoid costs.
- `verification.json`: independent serialized-data numerical audit, all40320 assignments per run, actual package TSP/refinement replay, source preservation and formula checks.
- `replication_verification.json`: two fresh full-pipeline replications using the isolated local helper modules, compared with saved scores/references.

Read numeric CSVs with `pd.read_csv(path, float_precision='round_trip')`. Original source rows are unit IDs1–144; internal array indices and cluster labels are zero-based. Raw source coordinates and their original text are preserved. Normalized values follow Population's axis-minimum/common-maximum-span map. No post-translation normalization occurs. KDE values in the CSV are in normalized density units; divide by scale² for original-coordinate density units. DI is unchanged by that common density conversion.

The public package DI pipeline does not implement several manuscript choices. `adapter.py` declares the required overrides explicitly, executes actual package OT/refinement/hardening and original scoring code, and does not patch source files. `ot_adapter.py` and `density_helper.py` contain source-identical reused adapter definitions from the previous audited experiment, isolated here to avoid a runtime dependency on older experiment folders; `adapter_provenance.json` records the copy checks. They are not replacements for the package. Original package sources and prior adapter files are hashed in the preselection record.

Reproduce in a **new code-only directory**, preserving all saved results. With the current repository environment and the fixed source paths in preselection.json:

```bash
mkdir /tmp/R1C1_fresh
cp di_reference_sensitivity_R1C1_20261013/*.py di_reference_sensitivity_R1C1_20261013/preselection.json /tmp/R1C1_fresh/
export PYTHONPATH=/home/bardia/projects/graphical-sampling
export MPLCONFIGDIR=/tmp/mpl-reference
.venv/bin/python /tmp/R1C1_fresh/run_experiment.py
.venv/bin/python /tmp/R1C1_fresh/analyze.py
.venv/bin/python /tmp/R1C1_fresh/make_figure.py
.venv/bin/python /tmp/R1C1_fresh/write_deliverables.py
.venv/bin/python /tmp/R1C1_fresh/verify.py
```

The existing `sources/` snapshots permit checking that the source paths resolve to the same data and manuscript. On another machine, preserve the original preselection record and explicitly remap those paths in a new reproducibility copy. No internet, new dataset, or extensive simulation is required.

The concise answer is conditional: official-run variation is modest in these cases; a lower-loss stress reference changes S2 DI by about0.1602 even relative to an official reference with an optimal TSP path. It qualifies at the stated5% native-OT/compactness tolerance; stricter rules and common-scale loss comparisons must be distinguished. No universal robustness or instability claim is supported.
