from dataclasses import dataclass
from abc import abstractmethod, ABC

from graphical_sampling.design import Design


@dataclass
class Criteria(ABC):
    @abstractmethod
    def evaluate(self, design: Design) -> float: ...

    def __call__(self, design: Design) -> float:
        return self.evaluate(design)


class VarNHT(Criteria):
    def evaluate(self, design: Design) -> float:
        return design.nht_variance


class MoranCriteria(Criteria):
    def evaluate(self, design: Design) -> float:
        expected_value, _ = design.moran
        return expected_value


class DensityDisparityCriteria(Criteria):
    def evaluate(self, design: Design) -> float:
        expected_value, _ = design.density_disparity
        return expected_value


class VoronoiCriteria(Criteria):
    def evaluate(self, design: Design) -> float:
        expected_value, _ = design.voronoi
        return expected_value


class LocalBalanceCriteria(Criteria):
    def evaluate(self, design: Design) -> float:
        expected_value, _ = design.local_balance
        return expected_value
