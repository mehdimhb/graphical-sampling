import pickle
from pathlib import Path
from typing import Iterator, Collection, Optional, Tuple, Dict, Any, List

import torch
from matplotlib import pyplot as plt

from .structs import MaxHeap, Sample


class Design:
    """
    A class representing a sampling design fully backed by PyTorch
    with cached statistics and vectorized computations.
    """

    # =========================================================================
    # Initialization and Setup
    # =========================================================================

    def __init__(
            self,
            inclusion: Optional[torch.Tensor] = None,
            variable: Optional[torch.Tensor] = None,
            permute: bool = False,
    ):
        # Initialize MaxHeap
        self.heap = MaxHeap[Sample]()

        # State tracking
        self.changes = 0
        self.permute = permute
        self._N: Optional[int] = None

        # Caches for expensive computations
        self._all_samples_and_prob: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._sip: Optional[torch.Tensor] = None
        self._nht_variance: Optional[float] = None
        self._indicator_matrix: Optional[torch.Tensor] = None

        self.inclusion: Optional[torch.Tensor] = None
        self.variable: Optional[torch.Tensor] = None

        if inclusion is not None:
            self.inclusion = torch.as_tensor(inclusion, dtype=torch.float32)
            self._N = len(self.inclusion)
            self._push_initial_design(self.inclusion)

        if variable is not None:
            self.variable = torch.as_tensor(variable, dtype=torch.float32)

    # =========================================================================
    # Core Logic (Push, Pull, Switch, Iterate)
    # =========================================================================

    def push(self, *args: Sample) -> None:
        """Pushes samples into the heap if they are not essentially zero probability."""
        for sample in args:
            if not sample.almost_zero():
                self.heap.push(sample)
        # Important: Any modification to the heap must invalidate caches
        self._flag_change()

    def pull(self, random: bool = False) -> Sample:
        """Retrieves a sample from the heap (either top priority or random)."""
        if random:
            return self.heap.random_pop()
        return self.heap.pop()

    def iterate(self, random_pull: bool = False, switch_coefficient: float = 0.5) -> None:
        """Performs one step of the sampling algorithm (pull 2, merge or switch)."""
        sample_a = self.pull(random_pull)
        sample_b = self.pull(random_pull)

        if sample_a.ids == sample_b.ids:
            # If samples contain identical units, merge their probabilities
            new_prob = sample_a.probability + sample_b.probability
            self.push(Sample(new_prob, sample_a.ids))
        else:
            # Otherwise, perform the switch/swap logic
            results = self._switch(sample_a, sample_b, switch_coefficient)
            self.push(*results)

        self.changes += 1
        # _flag_change is called inside push, so redundant here, but harmless.

    def merge_identical(self) -> None:
        """Consolidates samples with identical IDs in the heap."""
        sample_map: Dict[frozenset, float] = {}

        for sample in self.heap:
            sample_map.setdefault(sample.ids, 0.0)
            sample_map[sample.ids] += float(sample.probability)

        # Rebuild heap with consolidated samples
        consolidated_samples = [
            Sample(length, ids) for ids, length in sample_map.items()
        ]
        self.heap = MaxHeap[Sample](initial_heap=consolidated_samples)
        self._flag_change()

    @staticmethod
    def _switch(
            sample_a: Sample,
            sample_b: Sample,
            coefficient: float = 0.5,
    ) -> Tuple[Sample, Sample, Sample, Sample]:
        """
        Swaps units between two samples based on the Cube method logic.
        """
        length = coefficient * min(sample_a.probability, sample_b.probability)

        # Identify unique units in each sample relative to the other
        diff_a = list(sample_a.ids - sample_b.ids)
        diff_b = list(sample_b.ids - sample_a.ids)

        # Select random unit from differences using Global Torch State
        idx_a = torch.randint(high=len(diff_a), size=(1,)).item()
        idx_b = torch.randint(high=len(diff_b), size=(1,)).item()

        n1 = diff_a[idx_a]
        n2 = diff_b[idx_b]

        return (
            Sample(length, sample_a.ids - {n1} | {n2}),
            Sample(sample_a.probability - length, sample_a.ids),
            Sample(length, sample_b.ids - {n2} | {n1}),
            Sample(sample_b.probability - length, sample_b.ids),
        )

    def _push_initial_design(self, inclusions: torch.Tensor) -> None:
        """Internal method to decompose inclusion vector into initial samples."""
        events: List[Tuple[float, str, int]] = []
        level: float = 0.0

        indices = torch.randperm(len(inclusions)) if self.permute else torch.arange(len(inclusions))

        for i in indices:
            i = int(i.item())
            p = inclusions[i].item()
            next_level = level + p

            if next_level < 1 - 1e-6:
                events.append((level, "start", i))
                events.append((next_level, "end", i))
                level = next_level
            elif next_level > 1 + 1e-6:
                events.append((level, "start", i))
                events.append((1.0, "end", i))
                events.append((0.0, "start", i))
                events.append((next_level - 1, "end", i))
                level = next_level - 1
            else:
                events.append((level, "start", i))
                events.append((1.0, "end", i))
                level = 0.0

        events.sort()
        active = set()
        last_point: float = 0.0

        for point, event_type, bar_index in events:
            if event_type == "start":
                active.add(bar_index)
            elif event_type == "end":
                if last_point != point:
                    self.push(Sample(round(point - last_point, 9), frozenset(active)))
                active.remove(bar_index)
            last_point = point

    def _flag_change(self):
        """Invalidates all cached statistics."""
        self._all_samples_and_prob = None
        self._sip = None
        self._nht_variance = None
        self._indicator_matrix = None

    # =========================================================================
    # Properties and Statistics (Vectorized)
    # =========================================================================

    @property
    def fip(self) -> torch.Tensor:
        """First-order inclusion probabilities."""
        return self.inclusion

    @property
    def all_samples_and_prob(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns tensors of all sample configurations (S x n) and probabilities (S)."""
        if self._all_samples_and_prob is not None:
            return self._all_samples_and_prob

        all_samples = []
        all_prob = []

        for sample in self.heap:
            all_samples.append(list(sample.ids))
            all_prob.append(sample.probability)

        # Convert to Tensor
        samples_tensor = torch.tensor(all_samples, dtype=torch.int64)
        probs_tensor = torch.tensor(all_prob, dtype=torch.float32)

        # Normalize probabilities
        if probs_tensor.numel() > 0:
            probs_tensor = probs_tensor * (1.0 / probs_tensor.sum())

        self._all_samples_and_prob = (samples_tensor, probs_tensor)
        return self._all_samples_and_prob

    def _get_indicator_matrix(self) -> torch.Tensor:
        """
        Returns a binary Indicator Matrix I of size (S x N).
        I[s, k] = 1 if unit k is in sample s, 0 otherwise.
        """
        if self._indicator_matrix is not None:
            return self._indicator_matrix

        samples, _ = self.all_samples_and_prob

        # Dimensions
        S = samples.shape[0]
        N = self.population_size()

        # Create indicator matrix
        # scatter_ writes 1.0 into the indices specified by 'samples'
        indicators = torch.zeros((S, N), dtype=torch.float32)
        indicators.scatter_(1, samples, 1.0)

        self._indicator_matrix = indicators
        return self._indicator_matrix

    @property
    def sip(self) -> torch.Tensor:
        """
        Second-order inclusion probabilities as an N x N Matrix.
        Calculated using vectorized matrix multiplication: I^T * diag(p) * I.
        """
        if self._sip is not None:
            return self._sip

        _, probs = self.all_samples_and_prob
        indicators = self._get_indicator_matrix()

        # Weighted indicators: Multiply each row (sample) by its probability
        # shape: (S, N)
        weighted_indicators = indicators * probs.unsqueeze(1)

        # Matrix Multiplication: (N x S) @ (S x N) -> (N x N)
        # Entry (i, j) sums (p_s * I_si * I_sj) over all samples s
        self._sip = weighted_indicators.T @ indicators

        return self._sip

    @property
    def nht_variance(self) -> float:
        """Calculates the variance of the Horvitz-Thompson estimator."""
        if self._nht_variance is not None:
            return self._nht_variance

        samples, samples_probs = self.all_samples_and_prob

        # Vectorized calculation
        nht_values = self.variable[samples] / self.inclusion[samples]
        nht_estimator = torch.sum(nht_values, dim=1)

        true_total = self.variable.sum()
        variance = torch.sum(((nht_estimator - true_total) ** 2) * samples_probs)

        self._nht_variance = variance.item()
        return self._nht_variance

    def conditional_next_probs(self, given_ids: Collection[int]) -> torch.Tensor:
        """
        Calculates P(next unit = j | given_ids in sample).
        Vectorized implementation removing loops over samples.
        """
        N = self.population_size()
        given_tensor = torch.as_tensor(list(given_ids), dtype=torch.long)

        _, probs = self.all_samples_and_prob
        indicators = self._get_indicator_matrix()

        # 1. Identify samples containing the given set
        # Sum indicators only for the given columns. If sum == len(given), it's a subset.
        if len(given_tensor) > 0:
            matches_count = indicators[:, given_tensor].sum(dim=1)
            valid_mask = (matches_count == len(given_tensor))
        else:
            valid_mask = torch.ones(len(probs), dtype=torch.bool)

        # 2. Filter probabilities and indicators
        valid_probs = probs[valid_mask]
        valid_indicators = indicators[valid_mask]

        denominator = valid_probs.sum()

        if denominator == 0:
            return torch.zeros(N, dtype=torch.float64)

        # 3. Calculate Numerator: Sum (prob * indicator) for valid samples
        # shape: (N,)
        numerator = (valid_indicators * valid_probs.unsqueeze(1)).sum(dim=0)

        # 4. Normalize
        cond_inclusion = numerator / denominator

        # 5. Zero out already known units (they are already in the sample, p=0 for "next")
        if len(given_tensor) > 0:
            cond_inclusion[given_tensor] = 0.0

        # 6. Re-normalize to get a distribution over remaining units
        total = cond_inclusion.sum()
        if total == 0.0:
            return torch.zeros(N, dtype=torch.float64)

        return (cond_inclusion / total).to(torch.float64)

    # =========================================================================
    # Helpers
    # =========================================================================

    def sample_size(self) -> int:
        """Returns the fixed sample size n."""
        try:
            first_sample = next(iter(self.heap))
        except StopIteration:
            raise ValueError("Empty design: cannot determine sample size.")
        return len(first_sample.ids)

    def population_size(self) -> int:
        """Returns the population size N."""
        if self._N is not None:
            return self._N

        max_id = -1
        for s in self.heap:
            if s.ids:
                max_id = max(max_id, max(s.ids))

        if max_id < 0:
            raise ValueError("Cannot infer population size from an empty design.")

        self._N = max_id + 1
        return self._N

    # =========================================================================
    # Serialization
    # =========================================================================

    def state_dict(self) -> Dict[str, Any]:
        return {
            "heap": [
                (float(s.probability), tuple(sorted(s.ids)))
                for s in self.heap
            ],
            "changes": self.changes,
            "_N": self._N,
            "inclusion": self.inclusion,
            "variable": self.variable,
            "permute": self.permute
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "Design":
        d = cls()
        samples = [Sample(prob, frozenset(ids)) for prob, ids in state["heap"]]

        d.heap = MaxHeap[Sample](initial_heap=samples)
        d.changes = state["changes"]
        d._N = state["_N"]
        d.inclusion = state["inclusion"]
        d.variable = state.get("variable")  # .get for backward compatibility
        d.permute = state.get("permute", False)

        d._flag_change()
        return d

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self.state_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "Design":
        path = Path(path)
        with path.open("rb") as f:
            state = pickle.load(f)
        return cls.from_state_dict(state)

    # =========================================================================
    # Magic Methods & Viz
    # =========================================================================

    def show(self) -> None:
        initial_level: float = 0.0
        for sample in self.heap:
            for i in sample.ids:
                plt.plot([i, i], [initial_level, initial_level + sample.probability])
            initial_level += sample.probability
        plt.grid(False)
        plt.show()

    def copy(self) -> "Design":
        new_design = Design(permute=self.permute)
        new_design.heap = self.heap.copy()
        new_design.changes = self.changes
        new_design._N = self._N

        if self.inclusion is not None:
            new_design.inclusion = self.inclusion.clone()
        if self.variable is not None:
            new_design.variable = self.variable.clone()

        # Cache is invalid in new copy by default (initialized to None)
        return new_design

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.heap)

    def __len__(self) -> int:
        return len(self.heap)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Design):
            return NotImplemented
        return self.heap == other.heap

    def __hash__(self) -> int:
        return hash(self.heap)
