# Method and reproduction

This isolated module evaluates the mathematical pipeline specified in the available October12 Measure snapshot: sample coordinates initialize the OT centroids; balanced entropic OT alternates Sinkhorn/Newton solutions and centroid updates with median cost normalization; preliminary labels are maximum allocations; the existing GFS15-restart nearest-neighbor/2-opt OTSP heuristic orders centroids; the existing within-cluster ordering is retained; exact EP cumulative overlaps determine hard memberships; medoids minimize squared-distance loss with the specified IP/ID tie rule; squared matching is solved exactly with a deterministic lexicographic tie rule. The existing Newton helper solves the same convex fixed-centroid entropic OT problem, without changing its objective.

The user's requested historical spherical KDE differs from the box kernel displayed in that available manuscript snapshot. That explicit kernel choice is documented, not silently treated as identical. KDE uses KernelDensity(kernel="tophat", bandwidth="scott",rtol=1e-4), with h=144^(-1/6). Both fits have144 points and therefore the same numerical h. Original density is evaluated at original units, and the second density at the corresponding internally translated units. The whole-population normalization is raw/11; the compact sample is not rescaled on its own.

Exact integer grid medoid/matching costs are proportional to squared normalized-coordinate costs, so they have the same mathematical optima while avoiding floating rounding of genuine geometric ties. Equal-IP quota overlaps are also evaluated exactly. No tolerance is used to merge nearby unequal OT allocations or medoid costs. The approximate OTSP heuristic is preserved; exhaustive8! path enumeration is diagnostic only and does not replace the saved route.

All3 shapes and3 shifts were fixed in config.json before the first fit. Figure selection prioritizes a verified changed geometry/partition if found, never DI magnitude/sign. None was found; the first prespecified pair is shown. All other outcomes are retained. Thus these deliberately chosen cases do not estimate a frequency of invariance failure.

The displayed matching costs (raw squared-grid units) are:

- Original: best 438, second 444.
- Shifted: best 170, second 176.

The independent verifier compares all40320 matching permutations, explicitly checks medoid ties and partitions, reconstructs moved coordinates and strict spherical counts, recomputes Dnet/eta/DI with scalar arithmetic, validates row/column masses and centroid residuals, and verifies additive OT dual stationarity. Reference partition comparisons use both maximum-overlap label alignment and direct pairwise co-membership matrices. The separate frozen-population control shifts saved moved coordinates and verifies KDE/DI invariance; it is a density-only control, not a new matching optimization. Full joint-coordinate and same-seed replay controls rerun the full reference independently.

From the project root:
```bash
# Saved-state verification only:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_concentrated_JSSAM_20261016/verify.py
# Rebuild figure and report from saved states only:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_concentrated_JSSAM_20261016/make_figure.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_concentrated_JSSAM_20261016/write_report.py
# Resume is idempotent and skips completed fits:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_concentrated_JSSAM_20261016/run_experiment.py
```

The12 primary fits and4 controls are new small calculations. No completed480-evaluation study, historical Monte Carlo simulation or previous visual study was rerun. No historical fixed-seed claim is made. All computations use one thread and one worker, below the approximate55% CPU policy. protected_before.json includes the original protected files and all files from the preceding visual study. manifest.json records source/dependency versions and saved-result hashes.
