# Main-branch update completed

The user explicitly authorized making the current correction/astronomy branch main, and then explicitly reaffirmed proceeding after the ten research-suite failures were disclosed. The requested update is complete. No package version was released and no force-push was used.

- Candidate: `codex/gfs-numerical-corrections-20261016`, `af36c9e3ff36f957243188ade01a1e01400f3709` (already pushed).
- Local/remote main and active checkout: `af36c9e3ff36f957243188ade01a1e01400f3709`.
- Newly created and pushed emergency backup: `backup/main-before-consolidation-20261016` at `dc2881af84a41e4f7658b6d21401e58470871b55`. The older `backup/main-before-di-20261016` also retains that original main commit.
- Fresh `git fetch origin --no-tags` succeeded. Main is an ancestor of candidate; divergence is 0 main-only / 16 candidate-only commits. Candidate includes numerical fixes, stable ordering, astronomy and earlier prescribed-SIP/MaxEnt work (549 changed paths). Do not claim it includes all historical branches or uncommitted research folders.
- No local tracked edits or package sources were changed. SHA checks are in state.json. Unrelated deleted notebook, modified result files and untracked research directories remain preserved locally and are not included in the candidate.

## Verification

A clean Git archive of the exact candidate was tested with the existing virtualenv. Imports were checked to come from the archive. Command: `python -m pytest tests prescribed_sip/tests -q -p no:cacheprovider`. Result: **257 passed, 10 failed, 20 plotting warnings**, 128.40 seconds. All failures are in prescribed_sip/tests; none is in the package/astronomy suite under tests/. Tests were restricted to one CPU and single-threaded libraries. No original manuscript Monte Carlo, completed 480-run sensitivity study or full astronomy study was rerun.

The ten failures comprise:

- One missing `osqp` dependency, declared in prescribed_sip/requirements.txt but absent from the current virtualenv.
- Seven missing-notebook failures: stale references to simulations/jupyters/GFS_prescribed_SIP_demo.ipynb or GFS_prescribed_SIP_demo_simple.ipynb. The repository has relocated notebook copies under prescribed_sip/notebooks and elsewhere; exact intended equivalence has not been established. Do not manufacture copies or regenerate research results just to satisfy tests.
- One comparability-audit failure caused by the absent notebook hash.
- One test demands two processes while the verification explicitly restricted affinity to one CPU; production notebook correctly reports one worker. Recheck that specific test with two allowed CPUs if continuing (still within the CPU budget).

Full failures: tests.log. Exact candidate, original main, preservation hashes and failed test names: state.json. The temporary clean archive path in state.json may not survive a new tool/session environment; recreate it with git archive before resuming checks. Do not rely on that temporary directory for any unique work.

## Completed integration and remaining limitations

Refreshed origin, confirmed the fast-forward relation, created and pushed the emergency backup first, and then ordinarily pushed the exact candidate SHA to origin/main. Advanced local main with an expected-old-SHA check and switched to main only after its committed tree equaled the existing branch. Verified complete dirty/untracked status and protected-file hashes remained unchanged. The numerical-corrections branch remains available; no branch was deleted.

Direct `git ls-remote` verification checks GitHub's main and emergency-backup SHAs. Local evidence is in state.json. The previously reported ten research-suite failures remain unresolved and must not be described as passing. No release is authorized.

The uncommitted research directories and local notebook/result edits are still present locally; they were not committed or uploaded by this integration. To inspect the changes relative to the emergency backup, run `git diff backup/main-before-consolidation-20261016..main`. Any future rollback should be separately authorized and preserve shared Git history and local uncommitted work; no rollback has been performed.
