# Codex Task: Investigating DI Reference-Configuration Sensitivity for JSSAM

## Scientific objective

We are revising our JSSAM manuscript, *Measuring Spatial Spread via n-Means Balanced Clustering*.

Reviewer 1's Comment 1 is:

> The sample-adaptive initialization and grid example clarify the construction, but do not establish whether alternative equally or nearly suitable references can materially change DI for the same observed sample. Please address this through a theoretical argument or a focused example, preferably using an irregular or clustered population. An extensive simulation study is not required.

We need a focused, rigorous computational investigation that directly addresses this concern.

The objective is **not to demonstrate that DI is stable at all costs**, but to determine scientifically how much DI varies when alternative reference configurations of comparable quality are obtained for the same observed sample.

## 1. Review the existing implementation first

Before running experiments:

1. Inspect the current Measure manuscript and its mathematical definition of DI.
2. Inspect our `graphical_sampling` package, including the relevant classes and functions.
3. Review the existing `Legacy DI Variability.html` report and any underlying experimental scripts.
4. Identify discrepancies between the manuscript and the actual implementations.

Important findings from our previous investigation:

- The legacy package's reported simulation results appear to use sample-initialized weighted k-means, while the manuscript describes an OT-based preliminary clustering procedure.
- Some runs reportedly produced \(n-1\) clusters because of floating-point truncation of the inclusion-probability total.
- Approximate TSP solutions can affect the reference configuration.
- Hard-cluster assignments may be sensitive to numerical ties, particularly in the Clustered population with \(n=32\).
- The previous investigation found potentially substantial DI sensitivity in certain scenarios.

Verify these issues independently.

**Do not mix different definitions or implementations of DI.** Our primary investigation must follow the current manuscript's algorithm.

Do not silently substitute the legacy implementation for the manuscript method. If modifications are needed to implement the manuscript faithfully, create an isolated experimental implementation and document the differences.

## 2. Selected experimental scenarios

Investigate four scenarios:

| Scenario | Population | Sample size | Inclusion probabilities |
|---|---|---|---|
| 1 | Clustered | 16 | EP |
| 2 | Clustered | 32 | EP |
| 3 | Meuse | 10 | UP |
| 4 | Swiss | 100 | UP |

Use the same population definitions and inclusion-probability specifications as the manuscript.

For every scenario:

- Select one fixed, reproducible, valid sample using a prespecified seed.
- Keep that exact sample fixed throughout all comparisons.
- Keep the population, inclusion probabilities, KDE settings, and all nonexperimental parameters fixed.
- Use a reproducible baseline reference configuration.
- Check that the number of constructed clusters equals the intended sample size.

Record the selected sample IDs and random seeds.

Do not select samples based on whether their results appear favorable. If the original problematic Clustered \(n=32\) sample from our previous work is available, investigate it as an additional explicitly identified diagnostic case, or use it as the prespecified sample with clear justification.

## 3. Three experiments per scenario

Perform up to **40 runs per experiment per scenario**, after a five-run pilot.

### Experiment A: TSP variability only

- Hold the OT solution fixed.
- Generate alternative TSP orderings using different reproducible random seeds and/or valid heuristic starts.
- Complete the hard-clustering, reference construction, and DI calculations.
- Measure the differences between the resulting DI values.
- Record the TSP path length and relative excess over the shortest path found.

This assesses the influence of approximate TSP solutions.

Where computationally practical, calculate an exact TSP solution as an additional benchmark. Do not claim a solution is globally optimal unless an exact algorithm establishes that fact.

### Experiment B: OT variability only

- Use alternative valid OT initializations or convergence paths.
- Apply the same deterministic TSP heuristic and tie-breaking rules to each resulting OT configuration.
- Calculate DI for each configuration.
- Record the OT objective value, convergence diagnostics, and relevant balance residuals.

All OT solutions must satisfy the prescribed convergence and feasibility criteria.

Avoid counting merely unconverged or invalid numerical solutions as legitimate alternative references.

### Experiment C: Combined variability

- Vary both the OT solution and approximate TSP ordering.
- Calculate DI and the resulting reference configurations.
- Record their objective values and other diagnostics.

This assesses the total sensitivity of the reference construction to these computational choices.

**Important:** Across Experiment B, the centroid locations may change, so the TSP optimization problem itself may change. Keeping TSP "fixed" means keeping its deterministic computational rule fixed, not necessarily preserving the same permutation.

## 4. The critical reviewer question: Nearly equivalent references

For every generated reference, record:

1. OT objective value.
2. TSP path length and an appropriate relative-quality diagnostic.
3. Hard-cluster assignments.
4. Medoid configuration.
5. DI, \(D_{\mathrm{net}}\), and \(\eta\).
6. Numerical convergence and feasibility diagnostics.

Identify reference configurations that are different but approximately equivalent in their optimization quality.

Investigate predefined quality thresholds, such as 0.1%, 1%, and 2% relative objective differences.

These are sensitivity thresholds, not assertions of global optimality.

When OT and TSP vary together, assess their qualities separately. Avoid treating an arbitrary combination of incomparable objective values as a single quality measure.

For each qualifying group of alternative references, calculate:

\[
\Delta_r=\mathrm{DI}_r-\mathrm{DI}_{0},
\]

where \(\mathrm{DI}_{0}\) is the fixed-sample baseline DI calculated using the verified manuscript implementation.

Report:

- Mean signed DI difference.
- Mean absolute DI difference.
- Standard deviation of DI differences.
- Maximum absolute DI difference.
- DI range.
- Number of sign reversals relative to baseline.
- Relative objective differences.
- Changes in cluster assignments and medoids.

If the baseline DI is near zero, flag sign changes separately because their practical interpretation may differ.

**Crucially, do not compare a single-sample DI directly with the manuscript's Monte Carlo mean DI.** They are different quantities.

If equally or nearly suitable configurations produce large differences in DI, report them explicitly rather than excluding them.

If reference sensitivity largely disappears after correcting numerical ties or implementation problems, demonstrate that distinction.

## 5. Numerical safeguards

Implement deterministic tie-breaking for:

- Equal or nearly equal hard-assignment memberships.
- Unit ordering.
- Medoid selection.
- Assignment/matching.
- Exact or approximate TSP path ties.

Avoid artificial changes in the hard partition caused solely by floating-point rounding or reversal of the same undirected TSP path.

Preserve genuine alternative solutions rather than forcing all solutions to be identical.

Check KDE bandwidth consistency, OT convergence, sample-adaptive initialization, inclusion-probability totals, and DI reproducibility.

Run basic unit tests before expensive experiments.

## 6. Computational budget: VERY IMPORTANT

My computer is also being used for other work.

**Target approximately 55% of available logical CPU capacity and do not intentionally exceed 60%.**

Requirements:

- Detect the number of available logical CPUs.
- Limit worker processes and CPU affinity appropriately.
- Prevent NumPy, BLAS, OpenMP, and similar libraries from oversubscribing the available CPUs.
- Prefer parallel execution across independent experiments, with appropriately limited threads per worker.
- Avoid excessive RAM consumption.
- Save intermediate results regularly.
- Make the experiment resumable after interruption.
- Start with five pilot runs per experiment to estimate cost and confirm correctness.
- If the Swiss \(n=100\) experiment proves disproportionately expensive, report its projected computational cost and use an optimized or reduced-run strategy rather than monopolizing the computer.

Do not launch unnecessary exhaustive searches.

## 7. Live progress indicator — REQUIRED

I want to monitor the progress throughout execution.

Implement a terminal progress display using `tqdm`, `rich`, or an equivalent library.

Display at least:

- Overall completion percentage, from **0% to 100%**.
- Current population/scenario.
- Current experiment (A, B, or C).
- Completed runs / planned runs.
- Elapsed time.
- Estimated remaining time, when meaningful.
- Number of active workers.
- Current CPU utilization, if available.

Example:

`Overall  [██████████░░░░░░░░░░] 50% | 240/480 runs`

`Scenario: Swiss n=100 | Experiment B | 18/40`

`CPU: 54% | Elapsed: ... | ETA: ...`

Progress must reflect genuinely completed computations, not artificially estimated percentages.

Also save a machine-readable `progress.json` containing the current state, so progress can be checked independently.

If pilot testing causes a revised total computational budget, update the denominator transparently.

## 8. Final outputs

Create a separate experimental-results directory containing:

1. **A concise scientific report** summarizing methodology, results, and limitations.
2. **One main table** suitable for adaptation into the JSSAM Appendix.
3. Detailed raw results in CSV format.
4. At least one figure illustrating DI sensitivity versus reference quality.
5. A reproducibility record: random seeds, parameters, software versions, and Git commit hashes.
6. A short proposed LaTeX subsection for the Appendix.
7. A short proposed manuscript paragraph acknowledging reference sensitivity.
8. A draft point-by-point response specifically addressing Reviewer 1's Comment 1.
9. An explanation of whether the findings also help resolve Comments 3 and 5.

Please distinguish:

- Computational instability and implementation errors.
- Dependence on genuinely different reference configurations.
- Stability of Monte Carlo averages versus stability for individual observed samples.

Do not modify the existing manuscript, bibliography, original package, or existing simulation results. Put all experimental code and generated files in a separate directory.

## 9. Execution strategy

**Stage 1:** Inspect code and manuscript, verify definitions, identify problems, and report the experimental plan.

**Stage 2:** Implement and validate the experimental procedure using five-run pilots.

**Stage 3:** Execute the approved full experiment, targeting 40 runs per scenario per experiment, subject to computational feasibility.

**Stage 4:** Analyze reference quality, DI variability, and the mathematical interpretation.

**Stage 5:** Produce the final report, Appendix-ready LaTeX, and reviewer-response draft.

Proceed through the stages unless you encounter a material methodological mismatch or a serious computational blocker. If either occurs, explain it before running expensive experiments.

**Scientific priority:** The most important output is whether two genuinely different but nearly equally suitable references can produce materially different DI values for exactly the same observed sample.

Be rigorous, transparent, and conservative in the conclusions. We want convincing evidence for the reviewer, not artificially favorable results.