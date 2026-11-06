# Exact TSP focused investigation plan

1. Reuse the same N144 population, UP vector, S1/S2, observed-center OT, squared medoids/matching, fixed box H, and original DI score routine. Freeze all settings/source hashes before new calculations.
2. Enumerate all8! open paths, quotient reversal equivalence, use80-digit Euclidean lengths, identify all optimal equivalence classes and specify deterministic spatial-lexicographic tie breaking. Check with higher precision and independent float64 enumeration. No heuristic or alternative initialization is introduced.
3. Recompute the entire OT/refinement/medoid/matching/density pipeline for each of ten varied RNG seeds per sample with exact TSP. Recompute every other oriented optimal path separately to assess reversal/tie consequences. Preserve all states and numerical outputs.
4. Compare only the20 prior official approximate runs, excluding stress initializations. Analyze path gaps, hard/fractional memberships, medoids, original DI/Dnet/eta, and both soft and hard compactness. Do not infer hard balance or better clustering from shorter TSP.
5. Produce a short Markdown report and one compact table. Skip a figure if these fully show the answer. Independently verify full states, deterministic repetitions, exact enumeration/ties and source preservation. No manuscript edits or reviewer-response draft.
6. Package all output/code in a ZIP and place clearly named copies next to each other directly on Desktop.
