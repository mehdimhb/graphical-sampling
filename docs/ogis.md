# Optimal Graphical Indirect Sampling

`graphical_sampling.indirect` represents a joint sampling distribution over two
linked populations A and B. Each supported strip is a matching of exactly `n`
admissible pairs: neither an A unit nor a B unit can appear twice. A B unit may
have many admissible A neighbors across the design.

Start with [the Run-All notebook](../simulations/jupyters/20_ogis_indirect_sampling.ipynb).
It has one input cell for both populations, FIPs, links, auxiliary and optional
study variables, and search parameters. The
[executed example](../simulations/jupyters/20_ogis_indirect_sampling_executed.ipynb)
and [exported results](../simulations/results/ogis_example/ogis_outputs) show the
default run. Use this feature branch's package code; the prototype's standalone
search is not required.

## Minimal use

```python
import numpy as np
from graphical_sampling.indirect import (
    OGISInputs, LinkedDesign, optimize_gbfs, export_results,
)

cfg = OGISInputs(
    pi_A=np.array([18, 12, 12, 6]) / 24,
    pi_B=np.array([13, 15, 7, 10, 3]) / 24,
    links=[[1, 1, 0, 0, 0], [1, 0, 1, 1, 0],
           [0, 1, 0, 1, 0], [0, 0, 1, 0, 1]],
    x_A=[3, 1, 3, .5], x_B=np.ones(5),
    y_A=None, y_B=None,
)
initial = LinkedDesign.initialize(cfg)
result = optimize_gbfs(initial, iterations=150, proposals=35, seed=2026)
best = result['best']
matched_pairs = best.draw(seed=2026)  # tuple of zero-based (A index, B index)
SA, SB = best.sip()
export_results(result, 'ogis_outputs')
```

FIPs must be finite, strictly positive and at most one, and both vectors must
sum to the same positive integer `n`. `links` must be binary with shape
`(len(pi_A), len(pi_B))`. Auxiliary vectors must be finite with matching lengths.
`names_A` and `names_B` are optional display labels; identity always follows array
position. `y_A` and `y_B` are independently optional finite study vectors used
only for simulation evaluation. Input arrays are copied and made read-only.

## Initialization and linked exchange

A sparse transportation LP finds nonnegative masses `lambda[i,k]` on admissible
links, with row sums `pi_A` and column sums `pi_B`. A feasible network must satisfy
the full flow constraints; having at least one neighbor per unit is insufficient.
Incompatible links and FIPs raise `ValueError` before optimization.

Augmented Birkhoff decomposition introduces dummy vertices and a zero
dummy/dummy block. Every positive-probability permutation then contains exactly
`n` real A/B pairs. Reconstructed link masses and marginals are validated.
The LP may choose any feasible solution. The manuscript's specific initial
distribution is therefore a separate fixture, not the initializer's output.

`LinkedDesign(cfg, states, fixed_lambda=None)` also accepts a user-provided
distribution. `states` maps tuples of `(i,k)` pairs to strip probabilities.
Unsorted pairs are canonicalized and equivalent support fragments aggregated;
negative/nonfinite probabilities, forbidden links and duplicate endpoints are
rejected. Exactly zero probabilities are omitted. No positive-probability strip
is intentionally filtered by size. Keys include both matching identities and
unrounded probabilities.

`design.exchange(u, v, e, f, fraction=.75)` swaps two **complete** linked pairs
`e` and `f` between supported matchings `u` and `v`, on equal probability mass
`fraction * min(p_u, p_v)`. It splits probability strips and merges identical
resulting matchings. It returns a new validated design, or `None` for a forbidden
or ineffective move. Duplicate A or B endpoints are rejected before construction.
The original design is unchanged. All link masses, both FIP vectors, and `n`
remain fixed, up to floating-point validation tolerances.

## NHT objective and actual package search

For auxiliary values `x_A`, each strip has total estimator
`Xhat_A = sum(x_A[i]/pi_A[i])`. Its exact design variance is
`VA = sum(p_d * (Xhat_A(d) - sum(x_A))**2)`; B is analogous.
`nht_variance(values_A, values_B)` computes both variances from full support.

The objective is `alpha*VA/VA0 + (1-alpha)*VB/VB0`, with `alpha` in `[0,1]`.
An exactly zero baseline uses unit scale for that population, so creating positive
variance is penalized. An endpoint weight of zero removes that population's
variance from scoring. Study values are never used in the objective.

`optimize_gbfs` instantiates the package's existing
`graphical_sampling.search.GreedyBestFirstSearch` and calls its `run` method.
Optional keyword-only constructor hooks select a linked key and linked neighbor
factory. The existing queue, closed set, top-k, reservoir, backtracking, and
pruning framework are shared. Existing two-argument constructor calls and
single-population order/exchange behavior remain the default. The small-frontier
restart range is bounded so a restart cannot raise an empty-range error.

The hooks are:

- `state_key(design)`: immutable, hashable snapshot for duplicate detection.
- `neighbor_factory(design, iteration)`: yields `(node_type, design)` pairs;
  replaces both legacy neighbor generators. Iterations here are zero-based.
- `candidate_callback(design, score, iteration)`: observes each evaluated
  candidate; initial states use zero and generated states use one-based steps.
- `rng`: optional local `random.Random`; default uses the existing global module.

Custom states should use `output_mode='objective'` or `'silent'` when invoking
`run` directly. Spatial diagnostics require the original spatial `Design` API.
The OGIS wrapper selects silent output and sequential scoring. It uses local
NumPy and Python RNGs and does not reset global random states.

GBFS compares improvements rounded to nine decimals. The wrapper reports the
actual reservoir result under that policy. `max_frontier` follows existing GBFS
pruning: once the queue exceeds twice that value, it is reduced to that value.
`backtrack_depth` and `stagnation_limit` control restarts. Candidate history is
deduplicated by complete probability-aware key in discovery order, independent
of frontier pruning or reservoir membership.

## Inspecting and exporting results

`result` contains `initial`, `best`, `initial_score`, `best_score`, the actual
`search` object, `history`, `candidate_designs` (`(score, design)` tuples),
`candidate_iterations`, `best_candidates`, `generated_candidates`, `alpha`,
`scales`, `seed`, and `settings`. `explored_states` is the number of closed keys currently
retained by GBFS, which periodically clears older keys; it is not a lifetime
expansion count.

`link_masses` returns a copy of lambda. `conditional_probabilities` returns
`lambda / pi_A[:,None]`, conditional on A selection. `sip()` returns both
second-order inclusion matrices, including the FIPs on their diagonals.
`sample_table()` lists every supported matching and probability; `draw()` selects
one matching using an integer seed or a NumPy Generator.

`candidate_table(result)` and `variance_comparison(result)` return pandas tables.
`plot_links`, `plot_design`, and `plot_results` show the bipartite network,
complete linked support and optimization progress. Figures show every supported
strip, although very thin strips may be visually indistinguishable.

`export_results(result, directory)` writes:

- Initial and final support CSVs, each with all supported joint sample pairs.
- Initial and final SIP matrices for A and B, lambda and conditional assignment CSVs.
- Auxiliary and available study-variable variance comparisons and GBFS progress.
- All discovered candidate scores and complete distributions in gzip JSONL.
- Input/run metadata and PNG/PDF figures.

JSONL records use zero-based edge indices and full numeric probabilities, allowing
reconstruction of every candidate with `LinkedDesign`. CSV sample labels are for
human inspection. Existing files of the same names are overwritten. **Support of
one design is not the set of all admissible designs.** Search history contains
only candidates actually discovered by this run.

## Exact example and limitations

The supplied manuscript fixture contains 24 equal segments. Exact rational
calculation gives `(27/4, 1823/325)` = `(6.75, 5.609230769230769)` before exchange,
and `(73/12, 10841/2275)` = `(6.083333333333333, 4.765274725274725)` afterwards.
Tests assert these with absolute tolerance `1e-12`; the prototype's loose,
slightly different expected B variances are not copied.

This is a research heuristic, not a global optimizer. The dense augmented matrix
has dimension `N_A+N_B-n`; support, neighbor validation and full candidate history
can grow expensive. There is no optimized local-delta scorer or disk streaming.
LP feasibility uses SciPy HiGHS with absolute tolerance `1e-10`; design validation
uses `1e-9` with no relative tolerance. Decomposition ignores final residuals at
machine precision scaled by matrix size. Extremely small masses or ill-conditioned
flows may raise a numerical error and require rescaling or higher precision.
Zero actual inclusion is always rejected for a positive prescribed FIP; errors
within the absolute tolerance may still be large relative to tiny positive FIPs.
Different SciPy versions can choose different feasible initial link masses.
Reproducibility is intended for identical inputs, ordering, seed and environment.

Validation commands:

```bash
python -m pytest tests/test_indirect_design.py tests/test_indirect_search.py tests/test_indirect_reporting.py
python -m pytest tests
python -m pytest
```

The bare-suite baseline at implementation start had 273 passes and nine failures
in `prescribed_sip`, due to missing demo notebook artifacts and the unavailable
optional `osqp` dependency. These are independent of OGIS and recorded in the PR.
