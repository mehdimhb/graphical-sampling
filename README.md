# Graphical Sampling Method - Python Package

The **Graphical Sampling Method**, introduced by Panahbehagh (2025), presents an innovative approach to **finite population sampling** based on a graphical representation of first-order inclusion probabilities (FIP).  
This framework enables users to visualize FIP as bars on a continuous line and adjust bar positions to explore a wide range of sampling designs while preserving or controlling second-order inclusion probabilities (SIP).

This Python package, `graphical_sampling`, provides:

- Tools for constructing graphical sampling designs,
- Evaluation criteria for comparing designs (e.g., VarNHT),
- **A Monte Carlo Tree Search (MCTS) engine** for design optimization.

---

## ✨ Features

- Create unequal-probability sampling designs with fixed FIP.
- Control, modify, and explore SIP using graphical operations.
- Perform local transformations on designs via switch operations.
- Optimize sampling designs using:
  - **Monte Carlo Tree Search (MCTS)** (core search engine)
- Built-in clustering and spatial sampling utilities.

---

## 📦 Installation

Install via pip:

```bash
pip install graphical-sampling
```

---

## 🚀 Basic Usage Example (MCTS Optimization)

```python
import graphical_sampling as gs
import numpy as np

# Random generator and dataset
rng = np.random.default_rng()
N = 50
x = rng.random(size=N)  # Auxiliary variable
n = 5

# Generate inclusion probabilities that sum to n
inclusion = rng.random(N)
inclusion *= n / inclusion.sum()

# Construct initial sampling design
initial_design = gs.Design(inclusion)

# Define evaluation criteria (e.g. VarNHT for NHT variance)
nht = gs.criteria.VarNHT(x, inclusion)

# Create MCTS search engine
mcts = gs.search.MCTS(
    initial_design,
    nht,
    switch_coefficient=1.0,   # same as in switch() method
)

print("Initial criteria value:", mcts.initial_criteria_value)

# Optimize sampling design
mcts.run(
    max_iterations=2000,         # number of MCTS iterations (playouts)
    max_children_per_node=10,    # branching factor
    num_changes=1,               # number of local switches in expansion
    rollout_depth=5,             # number of random steps in simulation
)

print("Best criteria value:", mcts.best_criteria_value)

# Visualize initial vs optimized designs
initial_design.show()
mcts.best_design.show()
```

---

## 📚 Modules Overview

```
graphical_sampling/
│
├── design.py                # Core GFS design (heap of sample segments)
├── criteria/                # Evaluation measures (e.g., VarNHT)
├── search/                  # Search algorithms (MCTS)
│    └── mcts.py
├── sampling/                # Spatial & probability-based sampling helpers
├── clustering/              # Balanced clustering algorithms
├── measure/                 # Density and spread scoring
├── random/                  # Random coordinate & probability generators
└── ...
```

---

## 📖 Reference

For full method details, see:

**Panahbehagh, B. (2025). Graphical Sampling Method.**

---

## 👤 Maintainers

- Bardia Panahbehagh – bardia.panah@gmail.com
- Mehdi Mohebbi – mehdi.mohebbi23@gmail.com
- AmirMohammad HosseiniNasab – awmirhn@gmail.com
- Mehdi Hosseini Moghadam – m.h.moghadam1996@gmail.com

---

Enjoy exploring and optimizing graphical sampling designs with MCTS! 🎯
