# ============================================================
# FILE:
# graphical_sampling/criteria/multi_objective.py
# ============================================================

import numpy as np


# ============================================================
# FAST NHT VARIANCE FOR ANY VARIABLE
# ============================================================

def nht_variance_for_variable(design, y):

    samples, probs = design.all_samples_and_probs
    pik = design.pop.inclusions

    ht = np.sum(
        y[samples] / pik[samples],
        axis=1
    )

    total = np.sum(y)

    return np.sum(
        probs * (ht - total) ** 2
    ).item()


class MultiObjectiveCriteria:

    def __init__(

        self,

        objectives=[
            'MI',
            'VAR',
        ],

        weights=[
            0.5,
            0.5,
        ],

        scales={
            'MI': 0.05,
            'VAR': 1000,
            'VI': 1.0,
            'DI': 1.0,
            'BI': 1.0,
        },

        auxiliary=None,
    ):

        self.objectives = objectives
        self.weights = weights
        self.scales = scales
        self.auxiliary = auxiliary

        if len(objectives) != len(weights):

            raise ValueError(
                "objectives and weights must have same length."
            )

    # ========================================================
    # MAIN EVALUATION
    # ========================================================

    def __call__(self, design):

        metric_map = {}

        # ----------------------------------------------------
        # MORAN
        # ----------------------------------------------------

        if 'MI' in self.objectives:

            metric_map['MI'] = design.moran[0]

        # ----------------------------------------------------
        # VARIANCE
        # ----------------------------------------------------

        if 'VAR' in self.objectives:

            if self.auxiliary is None:

                design._nht_variance = None
                metric_map['VAR'] = design.nht_variance

            else:

                metric_map['VAR'] = nht_variance_for_variable(
                    design,
                    self.auxiliary
                )

        # ----------------------------------------------------
        # VORONOI
        # ----------------------------------------------------

        if 'VI' in self.objectives:

            metric_map['VI'] = design.voronoi[0]

        # ----------------------------------------------------
        # DENSITY DISPARITY
        # ----------------------------------------------------

        if 'DI' in self.objectives:

            metric_map['DI'] = design.density_disparity[0]

        # ----------------------------------------------------
        # LOCAL BALANCE
        # ----------------------------------------------------

        if 'BI' in self.objectives:

            metric_map['BI'] = design.local_balance[0]

        # ====================================================
        # COMBINED OBJECTIVE
        # ====================================================

        total = 0.0

        for obj, weight in zip(self.objectives, self.weights):

            if obj not in metric_map:

                raise ValueError(
                    f"Unknown objective: {obj}"
                )

            scale = self.scales.get(obj, 1.0)

            total += weight * (
                metric_map[obj] / (scale + 1e-12)
            )

        return total

    # ========================================================
    # RETURN ONLY REQUESTED METRICS
    # ========================================================

    def evaluate_selected(self, design):

        out = {}

        if 'MI' in self.objectives:

            out['MI'] = design.moran[0]

        if 'VAR' in self.objectives:

            if self.auxiliary is None:

                design._nht_variance = None
                out['VAR'] = design.nht_variance

            else:

                out['VAR'] = nht_variance_for_variable(
                    design,
                    self.auxiliary
                )

        if 'VI' in self.objectives:

            out['VI'] = design.voronoi[0]

        if 'DI' in self.objectives:

            out['DI'] = design.density_disparity[0]

        if 'BI' in self.objectives:

            out['BI'] = design.local_balance[0]

        return out

    # ========================================================
    # RETURN ALL METRICS ONLY IF YOU REALLY NEED THEM
    # ========================================================

    def evaluate_all(self, design):

        out = {}

        out['MI'] = design.moran[0]

        if self.auxiliary is None:

            design._nht_variance = None
            out['VAR'] = design.nht_variance

        else:

            out['VAR'] = nht_variance_for_variable(
                design,
                self.auxiliary
            )

        out['VI'] = design.voronoi[0]
        out['DI'] = design.density_disparity[0]
        out['BI'] = design.local_balance[0]

        return out