# Final historical implementation audit plan

Specification: user attachment d609cbc1-fda9-4dcb-a6ad-eb863a75fbb5/Pasted text.txt, 17 October 2026.

Goal: identify replication counts, assignment metric and KDE implementation associated with the manuscript's retained numerical results; distinguish confirmed provenance from reconstructed mechanisms and missing execution evidence.

Constraints: no Monte Carlo or repeat of the completed paired kernel experiment; no original manuscript, bibliography, reviewer response, package, dataset or saved-result edits. Only small deterministic mathematical diagnostics. Proposed replacements remain separate and preserve RevThree and RevFive semantics. No commit or push. Parallelize independent evidence tasks; use at least half the available CPU cores for worthwhile parallel CPU work.

Tasks:
- [x] Trace requested/completed/statistics-used replication counts across original and medoid-update outputs, notebooks and Git history; produce setting-level evidence and unresolved cases.
- [x] Trace ordinary versus squared matching in historical modules and notebook overrides; associate methods with saved summaries and manuscript figures.
- [x] Search for authoritative revision-colored manuscript/response sources. Latest RevThree/RevFive source not recovered; explicitly limit exact quotations to the designated 12-October snapshot and leave actual wrappers untouched. Prepare separate corrections.
- [x] Recheck historical top-hat evidence independently of numerical closeness; use saved paired results read-only.
- [x] Verify a small explicit Euclidean-assignment counterexample under uniform sample translation; prove squared-cost permutation independence and state sufficient conditional DI invariance precisely.
- [x] Integrate a concise A–E report, separate proposed LaTeX corrections, deterministic diagnostic script/output, source anchors, preservation checks and a complete handoff.

Evidence rules: distinguish requested sample count, saved completed rows, number passed to score/statistics and merely printed Iterations metadata. Do not convert absent per-replication logs into proof of historical execution. Do not label an older local manuscript current when revision-colored snapshots differ.
