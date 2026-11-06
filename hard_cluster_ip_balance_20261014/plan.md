# Hard-cluster IP balance experiment plan

**Goal:** Evaluate ordered hard IPnM balance in all five Measure populations without calculating DI or editing previous work.
**Architecture:** Independent, sample-initialized OT cases, frozen couplings, fixed documented approximate TSP, corrected exact overlap hardening. A monitored multiprocessing scheduler controls total machine CPU load and records incomplete cases.
**Tech stack:** Existing Python environment, NumPy/SciPy/Numba, pandas, matplotlib, psutil.
**Spec:** User's Reviewer 1 Comment 4 task, 2026-10-14.

- [x] Preserve exact population files, manuscript, relevant probability construction evidence, package sources, OT/TSP/overlap adapters and input hashes.
- [x] Audit every EP/UP vector, including unprocessed Swiss Area and simple proportional normalization. Primary UP follows existing Measure code: prob/Cu/clipped Area and the existing certainty-unit helper, explicitly distinguishing it from literal proportional weights.
- [x] Verify metric calculations and empty-cluster reporting with independent hand-computable cases.
- [x] Pilot one sample. Reuse three existing clustered EP n16/n32 sample/coupling states. For other cases preselect three samples with fixed seeds, SRS for EP and randomized pivotal sampling for UP with prescribed first-order probabilities.
- [x] Compute first replicate at every setting, then additional replicates where feasible. At n<=20, attempt all three despite an individual first-sample failure; at n>20, first-sample nonconvergence stops extra work for that setting. Cap each OT case at 240 process CPU seconds; no unconverged solutions enter primary summaries. Existing epsilon=.01, marginal tolerance1e-10, outer tolerance1e-7, fixed-point check2e-7, 2000 outer iteration limit.
- [x] Start eight single-threaded workers on 16 logical CPUs; adjust toward 50–55% measured machine CPU, pause/stop experiment workers at 58–60% or high memory. Log CPU/memory every two seconds. Preserve responsiveness with lowered process priority.
- [x] Independently verify memberships, mass, bound, convergence, reproducibility and preserved-source hashes. Produce 60-row setting summary, five-population compact table, cluster/run CSVs, two PNG/PDF figures, Markdown report and one self-contained ZIP.

Rulings: This is a standalone additive scientific experiment, with no package implementation changes or branch integration. Execute directly under the user-authorized scope, without an extra plan approval gate. No DI/KDE routine is invoked. Strictly invalid proportional UP vectors are reported and never passed to OT; the original Measure helper's adjusted vectors have separate explicit provenance.

Progress: Population/code/manuscript provenance captured (62 protected artifacts); 180 initial samples fixed and validated; metrics/controller/cache-identity regression checks pass (5 tests); pilot completed 3.46 CPU seconds; first-replicate batch is ongoing under resource_monitor.csv. Independent review findings resolved and documented in CODE_REVIEW.md. Replicate-phase scheduler uses persistent selective suspension (resource_monitor_v2.csv); default experiment phase is first.

Final progress: All first replicates and all 78 feasibility-selected additional replicates attempted (138 attempts total). 106 converged runs across 39/60 settings, 1720 clusters, 30 settings with three converged replicates. 32 attempted failures and 42 intentionally unattempted expensive-setting replicates are explicit. Six regression tests passed; 106 independent exact assignment/mass/bound/threshold/OT audits passed; all 62 protected originals and the copied populations/package retain hashes; four full repeats reproduce all 11 arrays bit for bit. No DI was computed and the manuscript was not edited. Final tables, figures, report and archive verification are recorded in delivery.json.
