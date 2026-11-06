# Codex Task — Verify the Historical DI Implementation for JSSAM

## Objective

We are preparing the minor revision of our JSSAM manuscript, *Measuring Spatial Spread via n-Means Balanced Clustering*.

A previous computational investigation suggested that the original manuscript's simulation results were produced using weighted k-means, whereas the current manuscript describes optimal transport (OT) for preliminary clustering.

**I am not convinced this discrepancy has been established. Please investigate it carefully before drawing conclusions.**

The paper has already undergone several review rounds. We want to avoid unnecessary changes to its simulation results, while ensuring complete scientific accuracy.

## 1. Establish what was actually used

Inspect:

- The current Measure manuscript.
- The original FullPaperGFSS manuscript, where relevant.
- The `graphical_sampling` package and its historical Git versions.
- The original simulation notebooks and scripts.
- The existing `DI_reference_sensitivity_JSSAM/AUDIT.md` and `REPORT.md`.
- Any saved simulation outputs or experiment configurations.

Determine precisely:

1. Which clustering algorithm actually generated the published DI simulation results?
2. Was weighted k-means used for preliminary initialization, as the principal clustering algorithm, or at another stage?
3. Was OT used elsewhere in the original computational pipeline?
4. Does the manuscript explicitly require OT for the DI reference construction?
5. Does the historical code reproduce the manuscript's method, or is there a genuine methodological discrepancy?

**Distinguish direct evidence from inference.** Do not conclude that the published results used a particular algorithm merely because one package version contains that algorithm.

Trace the actual execution path and historical provenance whenever possible.

## 2. Assess the consequences

If a genuine discrepancy is confirmed, determine whether it affects:

- The conceptual definition of DI.
- Sample-adaptive reference construction.
- Numerical DI values.
- Comparisons between sampling designs.
- The paper's main scientific conclusions.

Distinguish differences in implementation from differences that materially change the findings.

First investigate existing evidence and saved outputs.

Only if necessary, run a **small, controlled comparison** between the two approaches using identical populations, inclusion probabilities, and fixed samples.

Prefer a few representative cases, including a clustered population and Meuse.

Do not repeat the full Monte Carlo study.

If the comparison indicates substantial differences, document them before recommending further experiments.

## 3. Check the previously reported n−1 issue

The earlier report suggested that floating-point truncation of the inclusion-probability sum sometimes produced n−1 clusters.

Verify:

- Whether this issue actually affected the original published calculations.
- Whether it was present in the historical implementation used for those calculations.
- Whether it affected DI values or the substantive conclusions.

Do not assume that this issue occurred simply because the current source code permits it.

## 4. Protect the manuscript and existing research

**Very important restrictions:**

- Do NOT modify the Measure manuscript.
- Do NOT modify the original graphical_sampling package.
- Do NOT alter existing figures, tables, numerical results, or Git history.
- Do NOT rerun the 480 completed reference-sensitivity evaluations.
- Do NOT automatically recommend repeating all manuscript simulations.
- Preserve all existing reviewer-driven revisions.

Work in a separate audit directory.

Keep computational usage low, targeting no more than approximately 55% of available CPU capacity if numerical checks are required.

## 5. Final report

Prepare a concise report named `DI_implementation_audit.md`.

Answer these questions explicitly:

**A. Is there a confirmed discrepancy between the manuscript and the software used to generate its published results?**

Answer: Yes / No / Not established.

Provide specific evidence, including filenames, functions, historical versions, and relevant manuscript passages.

**B. If confirmed, is the discrepancy scientifically material?**

Distinguish effects on individual DI values from effects on the main conclusions.

**C. Can the existing manuscript results be retained?**

Explain under what conditions this would be scientifically defensible.

**D. What is the minimum necessary correction, if any?**

Prefer the smallest accurate correction that preserves the manuscript's existing contributions.

If the evidence establishes a material problem, state it clearly. Do not conceal or minimize it merely to protect the submission.

**E. Is any additional simulation genuinely necessary?**

Recommend additional computation only if the existing evidence is insufficient.

## Final instruction

Our goal is to resolve a specific uncertainty, **not to reopen the entire manuscript or redesign DI**.

Please investigate the evidence first, perform only necessary targeted checks, and give a concise scientific recommendation before proposing any manuscript changes.