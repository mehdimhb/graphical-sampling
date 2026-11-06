Investigation completed without rerunning paper-design optimization or changing manuscripts,
saved designs or historical results. The only production change is the
operator/comment at `graphical_sampling/order.py:860–862`: `np.tile` becomes
`np.repeat`. Source backups, exact patch, regression tests and audit outputs
accompany this report. All 208 original pickle files retain their SHA256 hashes.

The essential n-means/GMS construction is valid for global sample size and
first-order inclusion probabilities. Border freezing is unnecessary for these
properties and insufficient for an unconditional one-per-original-cluster
claim. A separate, genuine bug in current split-order construction was fixed.

| Construction/evidence | Conclusion |
|---|---|
| Full original-unit permutation, lengths pi_i<=1, total integer n | Exactly n distinct units, FIP pi_i, in exact arithmetic |
| All 45 distinct retained NMS/GMS-labelled designs | Adjacent pieces coalesce to one full bar per unit; all N IDs present; maximum per-unit share error 1.49e-14 |
| Pre-patch whole-order tiling, q=num_splits | With d=gcd(n,q), gives n/d distinct units and FIP pi_i/d, in exact arithmetic |
| Corrected adjacent subdivisions | Equivalent to the unsplit full-unit ordering for every q |
| Original fractional clusters after arbitrary reordering/exchanges | One selected unit/contribution per initial cluster is not a general guarantee |

The guarantee requires admissible probabilities and the stated GFS/systematic
sampling rule; a permutation alone is not sufficient without these conditions.
Literal saved arrays often have more than N rows because they retain fractional
pieces. The rigorous statement is equivalence after coalescing adjacent pieces,
not that every saved array literally has N rows.

Pipeline and historical distinctions:

1. `graphical_sampling/clustering/fip_balanced_nmeans.py:405–412,851–926` forms
   clusters by cutting cumulative IP mass at quota boundaries and records
   fractional border shares. `fit_zones:451–486` incorporates these shares in
   zones, then clears floor/ceil markers. Clearing markers removes the source
   of `fixed_ids`; it does not remove original population probabilities.
2. `graphical_sampling/order.py:831–868` uses zone IDs, ignores their fractional
   shares, keeps first appearances, appends missing population IDs, and gives
   each unit its full mass. This changes cluster attribution while preserving
   the global permutation invariant. `change:878–937` swaps units and zones
   and rebuilds the order; it does not check `fixed_ids` during these swaps.
   A historical border stored only as floor/ceil, outside the zone arrays, is
   implicitly left in place by within-zone swaps. After `fit_zones` embeds it
   in a zone and clears those markers, that implicit protection is lost.
3. `graphical_sampling/search/gbfs.py:95–117` constructs actual exchange and
   order neighbors. Order neighbors rebuild GFS; exchange neighbors copy the
   current probability heaps and use `Design.exchange:157–223`. Equal-mass
   exchanges preserve global FIPs/size. Its `fixed_ids` exclusion is in
   `design.py:203–204`, separate from `Order.change`.
4. `graphical_sampling/design.py:397–429` flattens to one cluster/zone and
   rebuilds GFS. It discards the original cluster hierarchy and any probability
   exchanges not encoded in the order. It is not a distribution-preserving
   operation after general mass exchanges. Final laws must be audited from
   retained heaps/caches, not reconstructed from ordering alone.
5. `graphical_sampling/design.py:101–146` projects complete bars modulo one
   and forms samples from active IDs; `:295–317` exposes the complete
   distribution; `:149–154` draws from it. Numerical tolerances and float32
   probabilities qualify exact-arithmetic statements.

The current notebook `simulations/jupyters/02_sampling_NMS_GMS_design_search.ipynb`
cell20 (In24), lines35–56, explicitly creates Regular EP n20, fits n-means,
then fits sweep zones (2,2). Cell8 defines the initial-order enumeration.
Archived `simulations/Store/jupyter/sampling_fighting_KMC.ipynb` cell13 (In217),
lines39–67, explicitly calls fit -> fit_zones -> Order.from_clusters ->
Design.from_order. These cells demonstrate the pipeline; neither is proof
that its present source generated every saved final pickle.

`fit_zones` marker clearing is tracked from commit `a4b654ecec` (2026-04-13),
whereas whole-order tiling/first-appearance collapse is tracked from
`f3dc818b31` (2026-04-21). Retained March Meuse/Swiss objects contain historical
border metadata and adjacent pieces. Applying current behavior retroactively
would misclassify their validity. Pre-patch rebuilding of Swiss UP q4 produced
sizes25,25,75 for n50,100,150, respectively; the retained designs themselves
have sizes50,100,150. Archived Meuse EP n30 rebuilt to15 before the fix.
All corrected rebuilds and all retained-order replays passed global checks.
These are diagnostic in-memory replays, not regenerated final designs.

Saved-design verification (full paths and all cases in `saved_design_audit.csv`):

| Population | N | n | IP | Saved NMS support | Saved GMS support | Evidence/stage |
|---|---:|---:|---|---:|---:|---|
| Meuse |155|10|UP|155|155|`simulations/best_designs/meuse/{initial,best}_design_df_meuse_10_pik_up.pkl`; best label does not establish post-polishing finality |
| Regular |1000|20|EP|50|50|`simulations/best_designs/robertson/initial_design_C20Z4_Regular1000_angle_snake_staggered.pkl` and `best_design_C20Z4_RegularPop1000.pkl`; final polishing lineage unresolved |
| Meuse |155|5 /10 /20|EP|31 /31 /31|51 /31 /352|Saved distributions, not inferred from N/n |
| Meuse |155|5 /20|UP|755 /155|155 /221|The n5 initial-labelled file differs from its order replay; filename alone does not prove untouched NMS provenance |
| Swiss |959|50 /100 /150|EP and UP|959 each|959 each|UP q4 adjacent pieces are valid |
| Regular |1000|50 /100 /200|EP|20 /10 /5|20 /10 /5|Several GMS objects are flattened |
| Aggregated |1000|20 /50 /100 /200|EP|50 /20 /10 /5|50 /20 /10 /5|Both n50 GMS variants have support20 |

All 45 raw and API distributions have fixed size n and distinct valid unit IDs.
Maximum absolute raw probability-sum error is 1.27e-7; maximum raw FIP error
is 5.064e-8. For cached/API probabilities these are 2.153e-7 and 5.159e-8.
Raw and API supports agree after duplicate merging; all retained raw fragments
are finite and strictly positive. No normalization or repair was written back.
The empty archived `simulations/Store/best_design/initial_design_df_meuse_5_pik_ep.pkl`
cannot be audited; its valid canonical counterpart is separate. Other archived
nonempty designs are included in the full CSV (Meuse3UP,30EP; Swiss20EP).

Testing the stronger cluster claim:

- A fractional counterexample uses pi=(.4,.4,.4,.8), with original clusters
  C0=(unit0:.4,unit1:.4,unit2:.2) and C1=(unit2:.2,unit3:.8).
  Reordering C0 as0,2,1 and collapsing full units gives order0,2,1,3.
  On U in[0,.2), units0 and1 are selected; neither belongs even fractionally
  to C1. Size2 and all FIPs are nevertheless correct. This is an actual
  package regression test at `tests/test_gfs_border_reordering.py:197`.
- In retained Meuse n20 EP GMS, a sample of probability0.007920788299255372
  contains no member of retained cluster label0. For n20 UP, a sample of
  probability0.011641863 contains no member of retained cluster label18.
  Exact selected indices (zero-based) and five retained-file witnesses are in
  `retained_cluster_counterexamples.json`. These disprove even a possible
  one-unit-per-soft-cluster assignment, not merely hard-label/weighted counts.
- Soft matching tests allow every strictly positive membership share.
  Total unmatchable probability mass is about0.485869 (Meuse20EP),0.130041
  (Meuse20UP),0.0967742 (Meuse5EP), and0.561590 (archive Meuse30EP), against
  their own retained cluster metadata. The n5 UP initial-labelled file also
  has unmatchable mass0.139673. Hard argmax counts and fraction-weighted counts
  are reported separately; neither is identified with actual fragment tags.
- Where retained ordered fragments form mass-one cluster blocks, the order
  replay has one contribution per block. This does not establish that final
  probability heaps after exchanges retain those tags. Comparisons with a
  separately saved initial clustering explicitly mark ancestry as unproven.
  Flattened designs cannot recover their original cluster tags from the best
  pickle alone.

Fixing border identities is insufficient: moving a zone moves its border
pieces; even keeping a unit at the same ordinal position does not keep its
cumulative-mass position under unequal-IP global permutations. One-per-original-
cluster contributions could instead be guaranteed by a sufficient geometric
condition: preserve mass-one cluster blocks and anchor cross-cluster tail/head
fragments at their block ends/starts through every reorder, with exchanges
also preserving cluster tags. General flattening/exchanges do not impose this.
That would restrict the methodology/search beyond the requested minimal fix.
No protected-border mechanism was added.

Published metrics and validity:

The patch changes only order construction. It does not run while loading or
scoring retained heaps/caches. Therefore it changes none of the retained
designs, their MI/VI values or existing table entries; no table requires
recomputation because of this patch. Fresh scoring of retained designs is
recorded separately from older summary rows in `saved_design_audit.json`.

There are pre-existing discrepancies requiring version/design reconciliation
before claiming all printed values have been independently reproduced:

| GMS case | Existing table MI /VI | Current saved design + current scorer MI /VI | Source |
|---|---|---|---|
| Meuse5 EP |-.40804027 /.07113559|-.44981446 /.08090469|`simulations/results/modified_summary_new_pivotal.csv:66`|
| Meuse5 UP |-.31267315 /.08349204|-.26395003 /.08349204|same:84|
| Meuse10 UP |-.34248649 /.08979412|-.33305455 /.08979412|same:85|
| Meuse20 UP |-.39293056 /.08653188|-.37646498 /.08653188|same:86|
| Swiss50 UP |-.19957442 /.12598884|-.15138771 /.12598884|same:151|
| Swiss100 UP |-.20304296 /.15312786|-.11831485 /.15312786|same:152|
| Swiss150 UP |-.19497967 /.16179727|-.01086058 /.15906757|same:153|
| Aggregated50 EP |-.59514941 /.10735000|main-.61001106 /.10735000; Var-MI-.59514941 /.10735000|`simulations/results/variance_outputs/variance_estimator_results_partial.csv:16`|

Regular20EP (-.36082239 /.04851920), Regular50EP (-.46522510 /.04067000),
and Aggregated20EP (-.43073555 /.08617361) agree with the retained variance
table to numerical precision. The old summary and current variance table
represent different analysis snapshots; these comparisons do not by themselves
establish the cause of every difference. If adopting current saved designs and
current scorers, the corresponding MI entries above, and VI entries for Meuse5EP
and Swiss150UP, need updating after resolving provenance. NMS comparisons are
also retained in the JSON. The full JSSAM manuscript/table numbering was not
available in the project, so exact manuscript table numbers cannot be assigned.
Do not attribute these differences to the split correction.

Remaining numerical/zone limitations, not historical execution claims:

`design.py:114–145,185–188,312–314` forces the final endpoint, snaps/skips tiny
intervals, drops tiny masses and casts/normalizes API probabilities. Exact
mathematical guarantees are implemented to numerical tolerance, as quantified
above. Separately, `fip_balanced_nmeans.py:872–917` can double-count a unit
spanning multiple fractional inner-zone quotas. A read-only helper diagnostic
with masses(.8,.1,.05,.05), three quotas1/3, gives masses(1/3,1.1333333,1/3);
see `fractional_quota_diagnostic.json`. Near-one rounding at line919 can also
erase tiny fractional remainders. Neither issue is evidence of a particular
historical execution. The full-unit collapse still guarantees global validity
under the proposition; no clustering rewrite or historical repair was made.

There is also an existing float32 CDF tail issue at `design.py:149–154`:21 of
45 cached distributions have `probs.cumsum()[-1] < 1`, so a uniform draw in
that tail would index beyond the support. The largest observed gap is1.377e-5;
file-specific endpoints are in `cdf_tail_diagnostic.json`. This is a separate
numerical sampler limitation, not evidence of an invalid retained sample or
of a historical execution failure. It was documented rather than expanding
the border/split patch into sampling or probability repair. Thus the proof
is an exact-arithmetic guarantee, not an unconditional claim about every
floating-point execution of the existing sampler.

Suggested manuscript replacement and a LaTeX proposition/proof are in
`proposition_and_wording.tex`. The smallest truthful change is to qualify
the original-cluster guarantee, retaining the n-means ordering and the valid
global sampling guarantees.

Verification: see `tests_red.log` (n10,q2 and n20,q4 fail before the patch),
`tests_targeted.log`, and `tests_package_final.log`. Independent review found
no critical/important issue with the minimal production correction. Package
baseline had233 passes and3 failures because the already-deleted notebook
`simulations/jupyters/17_graphical_maxent_die.ipynb` is missing; repository-root
collection also has3 unrelated module/import collisions in existing audit
directories. Final package run:359 passed (including all126 new regression
checks),3 unchanged missing-notebook failures,7 warnings;241.64 seconds.
