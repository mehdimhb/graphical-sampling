````markdown
# graphical-sampling

`graphical-sampling` is a Python package for finite-population sampling, with a particular focus on graphical sampling designs, unequal inclusion probabilities, and spatially well-spread samples.

The package implements the Graphical Finite-Population Sampling (GFS) framework and its spatial extensions, including probability-balanced `n`-means clustering, nested spatial ordering, and intelligent search procedures for improving spatial spread while preserving prescribed first-order inclusion probabilities.

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
  - exact Narain--Horvitz--Thompson variance when the response variable is supplied.

- Build probability-balanced spatial clusters using FIP-balanced `n`-means.
- Create nested cluster-zone structures for spatial sampling.
- Evaluate spatial spread using indices such as:
  - Moran-type spatial balance,
  - Voronoi-based spread,
  - Density Disparity Index,
  - local balance measures.

- **Improve sampling designs using advanced intelligent search procedures, including Greedy Best-First Search, Memory-Bounded A\* Search, and Adaptive Multi-Start Monte Carlo Tree Search (MCTS).**

---

## Installation

Install the package from PyPI:

```bash
pip install graphical-sampling
```
````

or install the development version from GitHub:

```bash
pip install git+[https://github.com/mehdimhb/graphical-sampling.git](https://github.com/mehdimhb/graphical-sampling.git)

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

## Intelligent Spatial Sampling & Optimization

The package includes search tools for improving a sampling design while preserving design validity. These methods modify the graphical order or exchange probability mass in a controlled way, and therefore maintain the prescribed inclusion probabilities.

Recent extensions have introduced highly optimized combinatorial search algorithms to minimize the Narain–Horvitz–Thompson (NHT) variance:

1. **A\* Search (Baseline):** A memory-bounded priority queue search that systematically expands nodes with the lowest variance.
2. **Multi-Start MCTS (Champion):** An adaptive algorithm that utilizes a diverse population of initial designs and teleportation mechanics to escape deep local optima, outperforming classical heuristic searches.

### Performance Comparison (NHT Variance Optimization)

Tested on a highly complex combinatorial population:

| Algorithm            | Final VarNHT   | Time (s) | Characteristics                                                                                                    |
| -------------------- | -------------- | -------- | ------------------------------------------------------------------------------------------------------------------ |
| **Initial Design**   | 227,845.50     | -        | Base randomized inclusion probabilities                                                                            |
| **A\* Search**       | 236,101.95     | ~70s     | Trapped in a severe local optimum (worse than initial) due to deterministic heuristic expansion and memory bounds. |
| **Multi-Start MCTS** | **133,265.65** | ~146s    | Successfully escaped local traps via adaptive rollouts, teleportation, and a multi-start population.               |

> **Conclusion:** The massive search space of finite-population sampling easily traps classical algorithms like A\* in deep local optima, sometimes yielding results worse than randomized designs. The Adaptive MCTS framework demonstrated remarkable superiority by halving the initial variance and consistently finding the global (or near-global) optimum.

A typical workflow for design optimization is:

1. Create a `Population`.
2. Build an initial design (or a population of designs) using GFS.
3. Choose a criterion, such as exact NHT variance via `operator.attrgetter('nht_variance')`.
4. Run an intelligent search algorithm (e.g., `MCTS` or `AStar`) to improve the design.
5. Use the optimized design for sampling and design-based inference.

---

## Citation

If you use `graphical-sampling`, please cite the software package. If you use the spatial clustering or intelligent spatial sampling methods, please also cite the corresponding methodological paper.

### Software citation

```bibtex
@software{graphical_sampling_2025,
  author = {Panahbehagh, Bardia and Mohebbi, Mehdi and HosseiniNasab, Amir Mohammad and Hosseini Moghadam, Mehdi},
  title = {graphical-sampling: A Python package for graphical finite-population and spatial sampling},
  year = {2025},
  url = {[https://github.com/mehdimhb/graphical-sampling](https://github.com/mehdimhb/graphical-sampling)},
  note = {Python package}
}

```

### Methodological papers

For the graphical finite-population sampling framework, cite:

```bibtex
@article{panahbehagh2026geometric,
  author = {Panahbehagh, Bardia},
  title = {Graphical Finite-Population Sampling},
  year = {2026},
  note = {Manuscript}
}

```

For the spatial sampling design, cite:

```bibtex
@article{panahbehagh2026intelligent,
  author = {Panahbehagh, Bardia and Mohebbi, Mehdi},
  title = {Intelligent n-Means Spatial Sampling},
  year = {2026},
  note = {Manuscript}
}

```

For the spatial spread measure, cite:

```bibtex
@article{panahbehagh2026spread,
  author = {Panahbehagh, Bardia and Mohebbi, Mehdi and HosseiniNasab, Amir Mohammad},
  title = {Measuring Spatial Spread via n-Means Balanced Clustering},
  year = {2026},
  note = {Manuscript}
}

```

Please replace the manuscript entries with the final journal citation once the papers are published.

---

## Maintainers

- Bardia Panahbehagh
- Mehdi Mohebbi
- Amir Mohammad HosseiniNasab
- Mehdi Hosseini Moghadam

---

## License

License information should be checked in the repository before redistribution.

```

```
