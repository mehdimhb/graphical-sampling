# ============================================================
# FILE:
# graphical_sampling/criteria/multi_objective.py
# ============================================================

import numpy as np


class MultiObjectiveCriteria:

    def __init__(

        self,

        # ====================================================
        # OBJECTIVES
        # ====================================================

        objectives=[

            'MI',
            'VAR',
        ],

        # ====================================================
        # WEIGHTS
        # ====================================================

        weights=[

            0.5,
            0.5,
        ],

        # ====================================================
        # SCALES
        # ====================================================

        scales={

            'MI': 0.05,

            'VAR': 1000,

            'VI': 1.0,

            'DI': 1.0,

            'BI': 1.0,
        },

        # ====================================================
        # AUXILIARY VARIABLE
        # ====================================================

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

            moran_val = design.moran[0]

            metric_map['MI'] = moran_val

        # ----------------------------------------------------
        # VARIANCE
        # ----------------------------------------------------

        if 'VAR' in self.objectives:

            # ================================================
            # MAIN VARIABLE
            # ================================================

            if self.auxiliary is None:

                design._nht_variance = None

                var_val = design.nht_variance

            # ================================================
            # AUXILIARY VARIABLE
            # ================================================

            else:

                samples, probs = design.all_samples_and_probs

                totals = []

                for s in samples:

                    y = self.auxiliary[s]

                    pik = design.pop.inclusions[s]

                    totals.append(
                        np.sum(y / pik)
                    )

                var_val = np.var(totals)

            metric_map['VAR'] = var_val

        # ----------------------------------------------------
        # VORONOI
        # ----------------------------------------------------

        if 'VI' in self.objectives:

            vi_val = design.voronoi[0]

            metric_map['VI'] = vi_val

        # ----------------------------------------------------
        # DENSITY DISPARITY
        # ----------------------------------------------------

        if 'DI' in self.objectives:

            di_val = design.density_disparity[0]

            metric_map['DI'] = di_val

        # ----------------------------------------------------
        # LOCAL BALANCE
        # ----------------------------------------------------

        if 'BI' in self.objectives:

            bi_val = design.local_balance[0]

            metric_map['BI'] = bi_val

        # ====================================================
        # COMBINED OBJECTIVE
        # ====================================================

        total = 0.0

        for obj, weight in zip(

            self.objectives,

            self.weights
        ):

            if obj not in metric_map:

                raise ValueError(
                    f"Unknown objective: {obj}"
                )

            scale = self.scales.get(obj, 1.0)

            standardized = (

                metric_map[obj]

                /

                (scale + 1e-12)
            )

            total += weight * standardized

        return total

    # ========================================================
    # RETURN ALL METRICS
    # ========================================================

    def evaluate_all(self, design):

        out = {}

        # ----------------------------------------------------
        # MORAN
        # ----------------------------------------------------

        out['MI'] = design.moran[0]

        # ----------------------------------------------------
        # VARIANCE
        # ----------------------------------------------------

        if self.auxiliary is None:

            design._nht_variance = None

            out['VAR'] = design.nht_variance

        else:

            samples, probs = design.all_samples_and_probs

            totals = []

            for s in samples:

                y = self.auxiliary[s]

                pik = design.pop.inclusions[s]

                totals.append(
                    np.sum(y / pik)
                )

            out['VAR'] = np.var(totals)

        # ----------------------------------------------------
        # OTHER INDICES
        # ----------------------------------------------------

        out['VI'] = design.voronoi[0]

        out['DI'] = design.density_disparity[0]

        out['BI'] = design.local_balance[0]

        return out