from dataclasses import dataclass
from abc import abstractmethod, ABC

import numpy as np

from ..design import Design
from ..new_design import NewDesign


@dataclass
class Criteria(ABC):
    auxiliary_variable: np.ndarray = None
    inclusion_probability: np.ndarray = None

    @abstractmethod
    def evaluate(self, design: Design | NewDesign) -> float: ...

    def __call__(self, design: Design | NewDesign) -> float:
        return self.evaluate(design)


class VarNHT(Criteria):
    def evaluate(self, design: Design) -> float:
        nht_estimator = np.array(
            [
                np.sum(
                    self.auxiliary_variable[list(sample.ids)]
                    / self.inclusion_probability[list(sample.ids)]
                )
                for sample in design
            ]
        )

        samples_probabilities = np.array([sample.prob for sample in design])

        variance_nht = (
            np.sum((nht_estimator**2) * samples_probabilities)
            - (np.sum(self.auxiliary_variable)) ** 2
        )

        return variance_nht


class MoranCriteria(Criteria):
    def evaluate(self, design: NewDesign) -> float:
        return design.kmeans.expected_moran_score()


class MoranWithPenaltyCriteria(Criteria):
    def evaluate(self, design: NewDesign) -> float:
        return design.kmeans.expected_moran_score() + (1-np.sum(design.kmeans.all_samples_probs))

