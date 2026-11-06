# DI cancellation investigation

User specification: 5,000 feasible samples, seed 12345, clustered N=144, n=4, equal inclusion probabilities; preserve all existing algorithms and data; investigate near cancellation and local sign flips.

1. Inspect Population normalization, DensityDisparity, FIPBalancedNMeans and available data.
2. Call the existing medoid DI pipeline; recover Dnet and eta from its returned densities with exactly its scaling operations. Generate SRS without replacement because the package RandomSampling uses replacement.
3. Record all random draws and search all one-unit replacements for the three closest high-eta samples of each sign. Separate random draws from targeted evidence.
4. Check the identity, repeatability, and clustering RNG sensitivity of illustrative pairs. Save numerical tables, plots, provenance and a concise interpretation.

No existing file is edited. No mathematical or algorithmic change, artificial coordinate, or invented density is used. Manuscript unavailable locally: certify the implemented definitions and flag manuscript comparison as pending rather than inventing agreement. New experimental scripts execute directly in the existing workspace, without a branch or publication action.
