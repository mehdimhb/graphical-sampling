# OGIS integration design

Implement the linked-design model from the supplied research prototype in
`graphical_sampling.indirect`, using the existing `GreedyBestFirstSearch.run`
loop. The user's explicit request to implement and open a review PR authorizes
execution. The supplied task is a requirements reference; the user's distinction
between the prototype driver and the package search takes precedence.

## Approach

Add optional state-key, neighbor-generation, candidate-observation, and local RNG
hooks to GBFS. With no hooks, the existing single-population path remains the
default. Linked neighbors bypass order changes and single-population exchanges;
they use the same priority queue, closed set, top-k, reservoir, and restart loop.
A separate standalone search would duplicate the framework and contradict the
request. Making linked designs imitate spatial `Design` objects would obscure
their matching constraints. Optional hooks keep those constraints explicit.

## Statistical contract

Inputs contain positive FIPs at most one, equal integer total n, binary links,
finite auxiliary vectors, optional finite study vectors and unit labels. Each
positive-probability strip is a matching of exactly n allowed pairs. All masses,
FIPs, and the initial link-mass matrix remain fixed during optimization.
Reject malformed probabilities and duplicate selections; never silently repair
them. Own and freeze input arrays and design distributions to keep search keys
safe. Numerical validation uses explicit absolute tolerances without relative
tolerance. Tiny positive mass must not be silently discarded.

A sparse transportation LP finds lambda. Augmented Birkhoff decomposition uses
dummy vertices and a zero dummy/dummy block to force n real edges per strip.
Check reconstructed mass and marginals. Initialization is arbitrary feasible,
not necessarily the manuscript's starting distribution.

An exchange moves two complete allowed linked pairs across equal masses in two
strips, rejects duplicate A or B endpoints, and splits/merges support. Auxiliary
NHT total variances define alpha*VA/VA0+(1-alpha)*VB/VB0. A zero baseline uses
an explicit unit scale; later positive variance is penalized, never ignored.
Study variables do not enter the objective. RNGs are local to each run.

## Outputs and validation

Expose support tables, lambda, conditional assignment probabilities, both SIP
matrices, exact variances, sampling, complete discovered-candidate distributions,
and chronological progress. Export CSV, gzip JSONL, and figures. Full candidate
history may consume substantial memory; GBFS remains a heuristic and does not
enumerate all admissible designs. Dense augmented decomposition limits scale.

The Run-All notebook imports package code and has one tagged user-input cell.
Show links, initial/final nested bars, variances, samples, SIPs and full history.
Distinguish the support of a distribution from the set of all possible designs.
Test the manuscript targets tightly: (27/4, 1823/325) before exchange and
(73/12, 10841/2275) after exchange of mass 1/24. Preserve all existing regression
tests, execute the notebook, and obtain an independent branch review. Deliver
only a feature branch and review PR; no merge or package publication.
