# Branch readiness assessment — 16 October 2026

Remote references were refreshed with `git fetch origin --no-tags`. The correction branch's local and remote tips are both `902ed8ef49edf4042ea8c30afefaa9aa974ad9ea`. Main remains `dc2881af84a41e4f7658b6d21401e58470871b55` locally and remotely.

## Findings

- The correction branch is the newest tip by recorded commit date among fetched origin branches and local branches. Commit dates do not establish completeness.
- Main is an ancestor: it has zero unique commits missing from the correction branch, which is seven commits ahead. Merging that branch at these SHAs preserves every current main commit.
- It does not contain seven newer-than-fork local astronomy commits (`codex/constrained-gfs-astronomy`), including graphical_sampling/constrained.py and its tests/notebook. Other historical/experimental branches also have divergent commits; this assessment does not assert those old alternatives ought to be merged or contain missing equivalent functionality.
- The full main-to-correction difference has 527 changed paths (518 additions, nine modifications), including prescribed-SIP studies, MaxEnt features/examples, plans and DI correction evidence. A full merge is broader than the approved DI patch.
- The DI correction commit alone changes 34 paths: three production files, three tests and 28 backup/report/evidence files. Its three production-file bases are identical on main and its actual parent, so the exact patch applies cleanly to main.
- The completed480-run DI reference-sensitivity directory, the historical implementation audit and other untracked/local edits are not contained in the pushed correction branch. Neither a main merge nor a backup of main will back up those uncommitted files.

## Fresh verification

A temporary export of the committed correction tree passed all105 committed package tests (16 plotting warnings). The earlier three missing-notebook failures were consequences of the notebook's local deletion; the committed notebook is present and those tests pass in the clean export. The local working-tree suite previously also included12 untracked exact-double-stratified notebook tests; therefore its totals differ from the clean committed suite. No original notebook/result was restored or modified in the user's checkout.

A second temporary export containing main plus exactly commit902ed8ef4's patch passed all68 DI correction/compatibility tests. Both exports asserted that imports came from their exported package, used the existing virtualenv and single-threaded libraries, and wrote only temporary test outputs. Existing reduced MaxEnt notebook validation ran in the first export; no original Monte Carlo or completed480 evaluation was rerun. These are package checks, not exhaustive validation of all research suites/features added by the preceding commits, nor a release certification.

Logs: clean_committed_tests.log and main_plus_correction_tests.log. Exact refs, divergence and export paths are recorded in assessment.json and main_plus_correction.json.

## Backup and recommendation

Created and pushed `backup/main-before-di-20261016` at the verified original main SHA `dc2881af84a41e4f7658b6d21401e58470871b55`. Main was not changed; the current working branch and unrelated dirty files were preserved.

Recommended integration: start a focused branch from the freshly verified main, cherry-pick902ed8ef4, and review that34-path change through a PR. Integrating all527 paths is a separate scope choice; the current branch already contains all main commits, so another main merge into it is unnecessary at these SHAs. Fetch/recheck main immediately before integration to account for concurrent changes.

After integration, an ordinary revert of the PR's merge/squash commit is preferable to force-resetting shared main. The backup gives the exact pre-change reference, but it does not validate the new code or preserve uncommitted data. No merge, package release or publication was performed. Publishing this backup branch only preserves an already-existing main commit.
