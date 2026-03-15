
# Geometric Sampling Method - Python Package

The Geometric Sampling Method, introduced by Panahbehagh (2024), presents an innovative approach to finite population sampling based on a unique geometric framework. This method allows researchers to visually depict first-order inclusion probabilities (FIP) as bars on a two-dimensional graph. By adjusting the positions of these bars, users can explore a wide range of sampling designs while controlling second-order inclusion probabilities (SIP).

This package, `geometric_sampling`, provides tools for implementing the Geometric Sampling Method and is available on PyPI.

---

## Features
- Create various unequal probability sampling designs with a fixed FIP.
- Control and explore second-order inclusion probabilities (SIP) through visual manipulation.
- Incorporate search algorithms, such as A* and Genetic Algorithm, to optimize sampling designs for specific needs (e.g., well-spread or optimal sampling).

## Installation

Install the package via pip:
```bash
pip install graphical_sampling
```

## How to Use

The package includes core classes and methods to facilitate sampling design creation and optimization. Below is a basic example demonstrating the API.

### Example Usage

```python
import graphical_sampling as gm
import numpy as np

# Set up random generator and sample data
rng = np.random.default_rng()
N = 50  # Population size
x = rng.random(size=N)  # Auxiliary variable
n = 5  # Sample size

# Generate initial inclusion probabilities
inclusion = rng.random(N)
inclusion *= n / inclusion.sum()

# Define initial sampling design and evaluation criteria
initial_design = gm.Design(inclusion)
nht = gm.criteria.VarNHT(x, inclusion)
astar = gm.search.AStar(initial_design, nht, switch_coefficient=1)

# Display initial criteria and design
print("Initial criteria value:", astar.initial_criteria_value)
astar.initial_designs.show()

# Run the A* search algorithm to optimize the design
astar.run(max_iterations=2000, num_new_nodes=10, max_open_set_size=10000, num_changes=1)

# Display results after optimization
print("Best criteria value:", astar.best_criteria_value)
astar.best_design.show()
```


## Maintainers

- Bardia Panahbehagh - [bardia.panah@gmail.com](mailto:bardia.panah@gmail.com)
- Mehdi Mohebbi - [mehdi.mohebbi23@gmail.com](mailto:mehdi.mohebbi23@gmail.com)
- AmirMohammad HosseiniNasab - [awmirhn@gmail.com](mailto:awmirhn@gmail.com)
- Mehdi Hosseini Moghadam - [m.h.moghadam1996@gmail.com](mailto:m.h.moghadam1996@gmail.com)

---

For more details, consult the official paper: Panahbehagh, B. (2024). Geometric Sampling Method.
