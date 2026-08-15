# MCTS simulations

Curated materials imported from the `mcts_graphical-sampling` branch.

## Contents

- `notebooks/`: A* vs. multi-start Monte Carlo tree search comparison notebooks.
- `populations/`: population CSVs used by the notebooks.
- `paper/`: ISC18 MCTS paper source and compiled PDF.

## Notes

The notebooks were originally at repository root and expected a top-level
`populations/` directory. They were moved under `simulations/mcts/notebooks/`,
and their population path was adjusted to `../populations`.

The notebooks also depend on MCTS/A* implementation modules from the
`mcts_graphical-sampling` branch. If they are run from `main`, those library
modules may need to be merged into `graphical_sampling/` as a separate code
change.
