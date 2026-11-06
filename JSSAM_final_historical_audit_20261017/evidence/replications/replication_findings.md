# Replication-count audit of the saved Measure results

Audit date: 2026-10-17. Authoritative historical snapshot: `e71d6935bfe824e72a7b722067a24775ed160610` (2026-06-01 02:57:37 +0330, “I submitted the aper”). All source was read from original Git blobs, as JSON/CSV/text or static pickle opcode buffers. No notebook or package code was imported or executed; no samples or index scores were generated. These counts concern the 162 DI settings in historical `simulations/jupyters/results/DI_nmedoids_selected_populations.csv`, not every experiment in the repository.

The saved results have **mixed count provenance**. The original Monte Carlo tables record **5000**; the later medoid DI update retains **106 successful setting outputs labelled 2000**, of which **104 have the exact final saved mean/SD pair**. Two final means differ from those outputs, and 32 additional MC settings have no comparable retained setting output. The 24 NMS/GMS DI settings use weighted enumeration, with **21,368 valid positive enumeration rows** across the designs. They are not 5000- or 2000-replication Monte Carlo summaries.

## Setting coverage

Every MC design listed below applies at each listed n. MAX under EP in the older auxiliary-index tables is the Rand/SRS row; MAX under UP is Maxe. Swiss has no WAV. The saved medoid source uses Maxe for its MAX generation under both EP and UP.

|Population|Probability|n|MC designs|MC settings|Saved output label 2000|Exact final mean/SD link|Mean discrepant|No comparable setting output|
|---|---|---|---|---:|---:|---:|---:|---:|
|Clustered|EP|4,8,16,32|MAX,SCP,LPV,WAV|16|16|16|0|0|
|Clustered|UP|4,8,16,32|MAX,SCP,LPV,WAV|16|0|0|0|16|
|Gridded|EP|4,8,16,32|MAX,SCP,LPV,WAV|16|16|16|0|0|
|Gridded|UP|4,8,16,32|MAX,SCP,LPV,WAV|16|16|16|0|0|
|Random|EP|4,8,16,32|MAX,SCP,LPV,WAV|16|16|15|1|0|
|Random|UP|4,8,16,32|MAX,SCP,LPV,WAV|16|0|0|0|16|
|Meuse|EP|5,10,20|MAX,SCP,LPV,WAV|12|12|11|1|0|
|Meuse|UP|5,10,20|MAX,SCP,LPV,WAV|12|12|12|0|0|
|Swiss|EP|50,100,150|MAX,SCP,LPV|9|9|9|0|0|
|Swiss|UP|50,100,150|MAX,SCP,LPV|9|9|9|0|0|
|Total||||138|106|104|2|32|

Machine-readable counterparts: `grouped_replication_counts.csv` and `per_setting_replication_ledger.csv`. The ledger has one row per final setting, original source paths, evidence line numbers, requested/recorded/inferred count distinctions, and finite-support evidence. “2000 inferred” is a source/dataflow conclusion for the 104 exact-linked pairs, not an independently retained `len(samples)` run log. The 32 unlogged settings are count-unknown; source configuration alone does not establish that those actual rows used 2000. The two discrepant final means also retain unresolved final-count provenance.

## What the medoid outputs establish

Historical notebook `simulations/jupyters/measure_calculation_nmedoids_clean.ipynb` cells 19 and 20 have execution counts 37 and 40. Cell 19 retains UP output for Gridded, Meuse and Swiss (37 rows); cell 20 retains EP output for all five populations (69 rows). Both sources set `SAMPLE_CNT_TEST=2000`, pass it as `num_samples`, compute `np.mean(d_scores)` and `np.std(d_scores, ddof=1)`, and assign `Iterations=SAMPLE_CNT_TEST`. Read `historical_medoid_notebook.txt` lines 954, 1031, 1049–1051 and 1493, 1575, 1593–1595. Cell 19 starts at line 944; cell 20 at 1483. The `Iterations` field is assigned the requested count; it is not obtained from an observed buffer length or a completed-job counter.

The two mismatches, after matching the exact population/probability/method/n/N key, are:

|Setting|Retained notebook mean|Final plotted CSV mean|SD in both|Notebook evidence|
|---|---:|---:|---:|---|
|Random EP MAX n=4|−0.089949|−0.089940|0.347306|cell 20, `historical_medoid_notebook.txt:2052`|
|Meuse EP LPV n=20|0.034492|0.034992|0.053606|cell 20, `historical_medoid_notebook.txt:2178`|

The static dataflow supports full requested-buffer scoring for a successful run: cell 10 allocates `(num_samples,n)` and iterates over `range(num_samples)`; cell 12 `validate_samples` rejects malformed dimensions, sample sizes, indices or within-row duplicates and returns the same array (lines 605–622). It does not drop, trim, retry or select successful rows. `score_samples` passes the whole sample array to the scorers (lines 625–635). The inherited historical `DensityDisparity.score` submits one task per supplied sample and returns every returned score (`historical_density_disparity.py:112–128`); there is no custom batch truncation, filtering, NaN dropping or exception-to-missing conversion. Joblib scheduling does not change the requested row count. A task error interrupts this path instead of returning a shortened score array.

Consequently, the exact-linked saved finite means and SDs are consistent with—and source-linked to—2000 scores per setting. We cannot independently inspect the actual MC buffers: no matching retained MC raw sample arrays, iteration-level results or actual completed-count logs were found in the audited historical Git tree (`historical_git_file_inventory.txt`). Existing retained tqdm strings are initial `0/2000` displays, not a final completion counter. The notebook's raw-save option is false and its separate full-results execution block is commented out.

## What 5000 belongs to

The older experiment already exists in ancestor `1f376e9fc1394feac1d02888b0ba0053fb48bceb` (2026-03-15). That commit introduces 36 tables under `simulations/results/equal_M=5000/` and `simulations/results/unequal_M=5000/`. All 138 MC setting rows selected for the final plotting population/design scope still have `ite=5000` in the submission snapshot. The initial notebook `simulations/jupyters/measurs.ipynb` cell 9 sets `sample_cnt=5000`; see `initial_March15_5000_notebook.txt:178`. Per-file first and last ancestral commits are saved in `original_5000_summary_file_history.csv`; exact original rows and source paths are in `original_5000_summary_rows.csv`.

The June 1 plotting notebook explicitly reads these older folders in cell 33 (`historical_plot_notebook.txt:2400`), selects original index means/SDs, averages duplicate keys, discards EP Maxe in favor of Rand renamed MAX, discards UP Rand, and clips LB mean/SD to [−1.5,1.5] (`:2467`, `:2474`, `:2478`, `:2487`, `:2490`). Cell 34 then merges the medoid table by population/probability/method/n/N and replaces **only Dm and Ds** (`:2869`, `:3007–3008`). Thus the old 5000-labelled tables still supply the auxiliary indices in the final master. All 138 MC VI/MI/LB pairs match their older tables after the documented LB clipping; four LB SDs become 1.5. This is directly verified numerically, not inferred merely from folder names.

The older generator allocates 5000 sample rows, scores them in full, appends exactly `range(sample_cnt)` score rows, aggregates by mean/std, and records the configured count (`historical_5000_notebook.txt:130`, `:227–244`, `:265–278`, `:301`). It does not log actual buffer lengths or nonmissing observation counts. The pandas aggregation could ignore missing values, so `ite=5000` does not independently establish an effective per-index count of 5000. The retained June 1 version of that cell actually ends with `ValueError: assignment destination is read-only` (`:321`); it cannot certify the old runs' completion.

A separate May 23 retained execution requests 5000 and prints one Random UP n16 Lopi table with `ite=5000`, then is interrupted during the n32 DI calculation (`May23_5000_notebook.txt:126`, `:471`, `:482`). Those numbers differ from the paper's old summary row, so this is additional 5000-run evidence, not a replacement provenance link for the paper results. Therefore 5000 is established as the older original experiment's recorded count, and also occurs in later exploratory runs; it is not evidence that the saved medoid DI update used 5000.

## The 24 NMS/GMS DI counts

Historical medoid notebook cell 22 (`historical_medoid_notebook.txt:2372`) obtains `samples,probs=design.all_samples_and_probs`, scores every enumeration row, normalizes probabilities, computes weighted mean and population SD, and returns `len(probs)` (`:2432–2454`). The original saved full-precision CSV `historical_saved_design_DI.csv` has all 24 means/SDs and Support counts. All 24 agree with the final plotting values rounded to six decimal places.

|Population|Probability|n|NMS enumeration rows|GMS enumeration rows|
|---|---|---:|---:|---:|
|Meuse|EP|5|34|51|
|Meuse|EP|10|31|31|
|Meuse|EP|20|31|352|
|Meuse|UP|5|755|155|
|Meuse|UP|10|155|155|
|Meuse|UP|20|155|221|
|Swiss|EP|50|959|959|
|Swiss|EP|100|959|959|
|Swiss|EP|150|959|959|
|Swiss|UP|50|3688|959|
|Swiss|UP|100|3536|959|
|Swiss|UP|150|3387|959|

To verify these counts without loading or executing a design, `inspect_static_design_buffers.py` uses `pickletools.genops` and `struct` to read the existing `_all_samples_and_probs` numeric cache in each original Git pickle. Each cached sample-array row count and probability count equals the saved Support. All 21,368 enumeration entries have exactly n distinct indices in the valid population bounds and a finite strictly positive probability; no entries are invalid, zero, negative or nonfinite. The minimum probability is 1.0301324e−9; sums range from 0.9999997847 to 1.0000002012 before the source helper's normalization. No DI scores were recomputed. `finite_design_support_counts.csv` records paths, Git blob hashes, SHA256, numeric counts and opcode byte offsets; per-design `*_cache_opcodes.txt` files show extracted array metadata.

“Support” here is the count of enumerated probability entries, not necessarily distinct unordered sample sets. Meuse EP NMS n5 has 34 entries but 31 distinct sample sets. Swiss UP NMS n50/n100/n150 has 3688/3536/3387 entries but 959 distinct sample sets each. All other designs have distinct sample-set count equal to enumeration count. Weighted statistics include all cached entries; repeats would be combined by summing probabilities if one represented the support as unique sets. None of these counts denotes MC replications.

Of the 24 NMS/GMS settings, 22 retain older auxiliary-index values from the old summary tables (some old rows carry a 5000 label; that label does not change the new DI enumeration count). Swiss UP n50 has older NMS/GMS rows erroneously labelled N=155 in `simulations/results/unequal_M=5000/summary_swiss_n=50_corr=0.7.csv`; the new medoid rows have N=959. Because N is part of the merge key, the latter are appended with missing auxiliary indices. The 166-row historical master therefore contains two obsolete N=155 Swiss rows and two Robertson rows beyond the 162-row DI plotting input. A claim that every index in every setting uses 5000, or that all 162 DI rows use 2000, is unsupported.

## Reproduction and evidence integrity

Run the following read-only parsers in order; writes stay within this evidence directory:

1. `python3 JSSAM_final_historical_audit_20261017/evidence/replications/extract_replication_audit.py`
2. `python3 JSSAM_final_historical_audit_20261017/evidence/replications/inspect_static_design_buffers.py`
3. `python3 JSSAM_final_historical_audit_20261017/evidence/replications/finalize_replication_evidence.py`

`source_manifest.json` records original CSV/notebook/package blob hashes and SHA256. No simulation, rescoring, notebook execution, package import or pickle unpickling is performed. Effective MC completion counts remain unverified where no original raw records exist; source-inferred and metadata-only counts are labelled separately in the ledger.
