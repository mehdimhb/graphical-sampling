# graphical-sampling

`graphical-sampling` is a Python package for finite-population sampling, with a particular focus on graphical sampling designs, unequal inclusion probabilities, and spatially well-spread samples.

The package implements the Graphical Finite-Population Sampling (GFS) framework and its extensions, including probability-balanced `n`-means clustering, nested spatial ordering, and intelligent combinatorial search procedures for variance minimization and spatial spread improvement while preserving prescribed first-order inclusion probabilities.

The package is designed for researchers and practitioners working in survey sampling, spatial statistics, environmental monitoring, ecological sampling, agricultural surveys, and related fields.

---

## Main Features

- Construct fixed-size sampling designs with prescribed first-order inclusion probabilities.
- Represent sampling designs through the graphical/bar construction of GFS.
- Draw samples from the resulting design.
- Compute design properties such as:
  - first-order inclusion probabilities,
  - second-order inclusion probabilities,
  - entropy and relative entropy,
  - exact Narain–Horvitz–Thompson (NHT) variance when the response variable is supplied.
- Build probability-balanced spatial clusters using FIP-balanced `n`-means.
- Create nested cluster-zone structures for spatial sampling.
- Evaluate spatial spread using indices such as:
  - Moran-type spatial balance,
  - Voronoi-based spread,
  - Density Disparity Index,
  - local balance measures.
- **Improve sampling designs using advanced intelligent search procedures, including Memory-Bounded Greedy Best-First Search (GBFS / A\*) and Adaptive Multi-Start Monte Carlo Tree Search (MCTS).**

---

## Installation

Install the package from PyPI:

```bash
pip install graphical-sampling
```

Or install the development version from GitHub:

```bash
pip install git+https://github.com/mehdimhb/graphical-sampling.git
```

Then import the package in Python:

```python
import graphical_sampling
```

Depending on the installation version, the main classes can also be imported directly from their submodules.

---

## Basic Example

The following example constructs a finite population with spatial coordinates, unequal inclusion probabilities, and a response variable. It then builds a graphical sampling design and draws samples from it.

```python
import numpy as np
from graphical_sampling.population import Population
from graphical_sampling.design import Design

# Reproducibility
rng = np.random.default_rng(123)

# Population size and sample size
N = 200
n = 20

# Spatial coordinates
coords = rng.random((N, 2))

# Unequal size measure, normalized internally to sum to n
weights = 0.5 + rng.random(N)

# Example response variable
y = coords[:, 0] + coords[:, 1] + rng.normal(scale=0.1, size=N)

# Create the finite population
pop = Population(
    coords=coords,
    inclusions=weights,
    variable=y,
    n=n
)

# Build a graphical sampling design
design = Design(population=pop)

# Draw five samples
samples = design.sample(num_samples=5)

print(samples)
print("Relative entropy:", design.relative_entropy)
print("NHT variance:", design.nht_variance)
```

---

## Spatial Sampling with FIP-Balanced `n`-Means

The package also provides probability-balanced spatial clustering. This is useful when the aim is to form compact spatial clusters whose total inclusion probabilities are controlled exactly.

```python
from graphical_sampling.population import Population
from graphical_sampling.design import Design
from graphical_sampling.order import Order
from graphical_sampling.clustering.fip_balanced_nmeans import FIPBalancedNMeans

# Fit FIP-balanced n-means clustering
fbn = FIPBalancedNMeans(
    n=n,
    n_init=20,
    init_clust_method="expanded"
)
fbn.fit(population=pop)

# Optionally divide each cluster into internal zones
fbn.fit_zones(
    num_zones=(2, 2),
    mode="sweep_xy"
)

# Build a spatial order from the cluster-zone structure
order = Order.from_clusters(
    population=pop,
    clusters=fbn.clusters,
    zone_strategy="snake",
    point_strategy="snake"
)

# Construct the corresponding spatial graphical design
spatial_design = Design.from_order(pop, order)

print("Moran index:", spatial_design.moran)
print("Voronoi index:", spatial_design.voronoi)
print("Density disparity:", spatial_design.density_disparity)
```

---

## Intelligent Sampling & Design Optimization

The package includes search tools for improving a sampling design while preserving design validity. These methods modify the graphical order or exchange bar segments in a controlled way, maintaining the prescribed first-order inclusion probabilities.

Recent extensions introduce combinatorial search algorithms to minimize the exact Narain–Horvitz–Thompson (NHT) variance:

1. **Memory-Bounded GBFS / A\* Search (Baseline):** A priority queue search that systematically expands designs with the lowest current variance criterion.
2. **Adaptive Multi-Start MCTS:** An adaptive tree search utilizing a diverse population of initial designs, reward-conditioned multi-scale mutations, heuristic rollouts with early stopping, and incumbent restart (teleportation) mechanics to escape local optima.

### Benchmark Summary on MU284 Dataset (18 Scenarios)

Evaluated across 18 scenarios on the filtered `MU284` population ($N=281$) under an equal budget of 4,000 exact variance evaluations (Relative Efficiency $\text{RE} = 100 \times \text{Var}_{\text{GBFS}} / \text{Var}_{\text{MCTS}}$):

| Variable Type                   | MCTS Wins (RE > 100%) | A\* (GBFS) Wins |  Median RE  |    RE Range    |
| :------------------------------ | :-------------------: | :-------------: | :---------: | :------------: |
| **Optimization Variable (Aux)** |      **16 / 18**      |     2 / 18      | **103.73%** | 96.5% – 112.7% |
| **Study Variable (S82)**        |      **13 / 18**      |     5 / 18      | **103.43%** | 94.7% – 111.6% |

A typical workflow for design optimization is:

1. Create a `Population` or initialize a `Design` with prescribed inclusion probabilities.
2. Build an initial population of designs using cumulative GFS and short greedy walks.
3. Choose an optimization criterion, such as exact NHT variance via `operator.attrgetter('nht_variance')`.
4. Run an intelligent search algorithm (`MCTS` or `AStar`) to optimize the design.
5. Use the optimized design for sampling and design-based inference.

---

## Citation

If you use `graphical-sampling`, please cite the software package. If you use the GFS framework, spatial clustering, or MCTS optimization methods, please also cite the corresponding methodological papers.

### Software citation

```bibtex
@software{graphical_sampling_2025,
  author = {Panahbehagh, Bardia and Moharrampour, Elahe and Mohebbi, Mehdi and  HosseiniNasab, Amir Mohammad and Hosseini Moghadam, Mehdi},
  title  = {graphical-sampling: A Python package for graphical finite-population and spatial sampling},
  year   = {2025},
  url    = {https://github.com/mehdimhb/graphical-sampling},
  note   = {Python package}
}
```

### Methodological papers

For the graphical finite-population sampling framework, cite:

```bibtex
@article{panahbehagh2026geometric,
  author = {Panahbehagh, Bardia},
  title  = {Graphical Finite-Population Sampling},
  year   = {2026},
  note   = {Manuscript}
}
```

For the adaptive MCTS design optimization, cite:

```bibtex
@inproceedings{panahbehagh2026mcts,
  author    = {Panahbehagh, Bardia and Moharrampour, Elahe and Mohebbi, Mehdi},
  title     = {Graphical Sampling with Adaptive Monte Carlo Tree Search},
  booktitle = {Proceedings of the 18th Iranian Statistical Conference (ISC18)},
  year      = {2026}
}
```

For the spatial sampling design, cite:

```bibtex
@article{panahbehagh2026intelligent,
  author = {Panahbehagh, Bardia and Mohebbi, Mehdi},
  title  = {Intelligent n-Means Spatial Sampling},
  year   = {2026},
  note   = {Manuscript}
}
```

For the spatial spread measure, cite:

```bibtex
@article{panahbehagh2026spread,
  author = {Panahbehagh, Bardia and Mohebbi, Mehdi and HosseiniNasab, Amir Mohammad},
  title  = {Measuring Spatial Spread via n-Means Balanced Clustering},
  year   = {2026},
  note   = {Manuscript}
}
```

Please replace the manuscript entries with the final journal citation once the papers are published.

---

## Maintainers

- Bardia Panahbehagh
- Elahe Moharrampour
- Mehdi Mohebbi
- Amir Mohammad HosseiniNasab
- Mehdi Hosseini Moghadam

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
