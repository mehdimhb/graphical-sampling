# Independent review and corrections

A fresh reviewer inspected the experiment, source provenance, exact hardening, aggregation, resource control, figures and reproduction workflow. No critical numerical issue was identified. Review was performed while computations were running; final coverage claims are checked separately by verify.py.

- Cached OT diagnostics carried an old case_id into new summary metadata. A full cached-pipeline regression test reproduced the mismatch. Whitelisting convergence fields preserves the design case_id; the original cached ID is namespaced as cached_source_case_id. Newly generated summaries were corrected, with raw couplings and assignments unchanged. Analysis and verification assert directory/design IDs. Cached objective/other irrelevant fields are excluded.
- First-phase CPU control lowered the target but temporarily paused all running workers, allowing oscillation and underutilization under competing load. The replicate-phase controller persistently suspends excess workers, maintains concurrency after a full pause and records runnable/suspended counts. Full pauses remain at the hard CPU/memory limits. Regression cases pin the expected controller response. Both monitoring logs are preserved.
- The original default all command bypassed the first-replicate feasibility stage. It was removed: the default is first, and additional replicates use the documented separate feasibility-controlled command.
- Protected-original verification now rejects missing inputs by default. Portable reproduction can explicitly allow unavailable originals, recording the unavailable list and declining a complete preservation claim.
- The reproduction instructions include repeat_check.py.

The six-check regression suite passed after the corrections. No original package, population, manuscript or prior-result file was changed. Final full-state numerical evidence and source hashes are recorded in verification.csv/json; fixed-seed repeats are recorded in repeatability.json.
