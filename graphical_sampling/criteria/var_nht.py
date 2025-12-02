import numpy as np

from ..design import Design
from .criteria import Criteria

from typing import Optional

class VarNHT(Criteria):
    def __call__(self, design: Design) -> float:
        N = len(self.auxiliary_variable)
        # Calculate estimators
        ones = np.ones(N)
        NHT_estimator_z = np.array([
            np.sum(self.auxiliary_variable[list(sample.ids)] /
                   self.inclusion_probability[list(sample.ids)])
            for sample in design
        ])
        NHT_estimator_z2 = np.array([
            np.sum(self.auxiliary_variable[list(sample.ids)]**2 /
                   self.inclusion_probability[list(sample.ids)])
            for sample in design
        ])
        NHT_estimator_ones = np.array([
            np.sum(ones[list(sample.ids)] /
                   self.inclusion_probability[list(sample.ids)])
            for sample in design
        ])
        NHT_estimator_y = np.array([
            np.sum(self.main_variable[list(sample.ids)] /
                   self.inclusion_probability[list(sample.ids)])
            for sample in design
        ])
        samples_probabilities = np.array([sample.probability for sample in design])
        var_NHT_z = np.sum((NHT_estimator_z - np.sum(self.auxiliary_variable)) ** 2 * samples_probabilities)
        var_NHT_z2 = np.sum((NHT_estimator_z2 - np.sum(self.auxiliary_variable**2)) ** 2 * samples_probabilities)
        var_NHT_ones = np.sum((NHT_estimator_ones - np.sum(ones)) ** 2 * samples_probabilities)
        E_NHT_estimator_z = np.sum(NHT_estimator_z *samples_probabilities)
        E_NHT_estimator_y = np.sum(NHT_estimator_y *samples_probabilities)

        # Calculate diagnostics
        if self.balance_method == 'origin':
            var_NHT = var_NHT_z
        elif self.balance_method =='linear':
            #print('the vars:', var_NHT_z, var_NHT_ones, var_NHT_z/E_NHT_estimator_z , var_NHT_ones/N, var_NHT_z/E_NHT_estimator_z + var_NHT_ones/N)
            var_NHT = var_NHT_z/np.abs(E_NHT_estimator_z) + var_NHT_ones/N
        elif self.balance_method == 'quadratic':
            var_NHT = var_NHT_z + var_NHT_ones + var_NHT_z2
                    
        var_NHT_y = np.sum((NHT_estimator_y - np.sum(self.main_variable)) ** 2 * samples_probabilities)

        # Store in attributes for later access
       
        self.var_NHT = var_NHT
        self.var_NHT_z = var_NHT_z
        self.var_NHT_y = var_NHT_y
        self.E_NHT_estimator_z = E_NHT_estimator_z
        self.E_NHT_estimator_y = E_NHT_estimator_y
        # Optionally store "all" as a dict or namedtuple if you want
        self.last_result = {
            'var_NHT': var_NHT,
            'var_NHT_z': var_NHT_z,
            'var_NHT_y': var_NHT_y,
            'E_NHT_estimator_z': E_NHT_estimator_z,
            'E_NHT_estimator_y': E_NHT_estimator_y,
        }
        # Return scalar value for optimization (main criterion)
        return var_NHT  # or whichever is the canonical value