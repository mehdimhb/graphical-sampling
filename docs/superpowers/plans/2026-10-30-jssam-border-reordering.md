# JSSAM border and reordering investigation

Scope: identify the mathematical invariant before changing code; preserve n-means,
the optimization objective, all saved designs, manuscript files and historical
outputs. User authorized a minimal correction after diagnosis. No optimization.

Invariant: a permutation of original units with complete masses pi_i <= 1 and
integer total n gives exactly n distinct units and prescribed first-order IPs
under systematic/GFS sampling. Adjacent subdivisions of an original bar are
equivalent. This does not imply one unit from each original fractional cluster.

Diagnosis/proposal communicated before implementation: current `_build_order`
tiles whole permutations for num_splits > 1. For d=gcd(n,num_splits)>1 this gives
n/d units with IP pi_i/d. Replace tiling by adjacent repetitions, consistent
with `Order.from_indices`. Do not add border protection: it is unnecessary for
global validity and insufficient for the stronger original-cluster claim.

1. Add failing behavioral tests for split orders, fractional borders, mutations,
   flattening, exchanges and the exact original-cluster counterexample. Exercise
   n=5/10/20, EP/UP and seeded clustering, plus saved NMS/GMS designs.
2. Observe split failures, change only tile to repeat, run targeted tests.
3. Audit all distinct retained NMS/GMS pickles, raw heaps and API probabilities,
   contiguous complete unit masses, original-cluster allocation when recoverable,
   current rebuild behavior and MI/VI comparisons. Keep outputs in the new audit
   directory. Verify historical file hashes afterward.
4. Run package tests and broader collection where feasible; obtain independent
   review. Record limitations and supply a LaTeX proposition and minimal wording.

Artifacts: `JSSAM_border_reordering_audit_20261030/` contains source backups,
manifests, audit outputs, exact patch, test logs and report. No original file
besides the diagnosed production line may be changed; new tests are allowed.

Ruling: user explicitly authorized the conditional local patch and backups;
work in place with source backups, without branch/index operations in read-only
`.git`. Preserve all pre-existing dirty files. No optimization or saved-design
overwrite. Separate numerical tolerances and older clustering limitations from
the demonstrated exact-arithmetic tiling error.
