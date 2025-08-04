import numpy as np

from ..design import Design
from .criteria import Criteria


class VarNHT(Criteria):
    def __call__(self, design: Design) -> float:
        nht_estimator = np.array(
            [
                np.sum(
                    self.auxiliary_variable[list(sample.ids)]
                    / self.inclusion_probability[list(sample.ids)][:, np.newaxis]  # Fix broadcasting
                )
                for sample in design
            ]
        )

        samples_probabilities = np.array([sample.probability for sample in design])

        variance_nht = (
            np.sum((nht_estimator**2) * samples_probabilities)
            - (np.sum(self.auxiliary_variable)) ** 2
        )

        return variance_nht
