from math import isclose
from typing import List, Optional, Tuple

import numpy as np

from geometric_sampling.design import DesignGenetic
from geometric_sampling.structs import Sample

EPSILON = 1e-12


class GeneticOptimizer:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def partition_design(
            self, fip_list: list[float], num_partitions: int
    ) -> tuple[dict[int, list[int]], set[int]]:
        """Partitions a list of FIPs into nearly equal sum partitions."""
        target = sum(fip_list) / num_partitions
        partitions, border_units = {}, set()
        current_partition, cumulative_sum = 0, 0.0
        partitions[current_partition] = []

        for index, fip in enumerate(fip_list):
            if (isclose(cumulative_sum + fip, target, abs_tol=1e-9)
                    or cumulative_sum + fip < target):
                partitions[current_partition].append(index)
                cumulative_sum += fip
            else:
                if not isclose(cumulative_sum, target, abs_tol=1e-9):
                    border_units.add(index)
                    cumulative_sum = fip - (target - cumulative_sum)
                    partitions[current_partition].append(index)
                    current_partition += 1
                    partitions[current_partition] = [index]
                    continue

                current_partition += 1
                partitions[current_partition] = []
                cumulative_sum = fip
                partitions[current_partition].append(index)

        return partitions, border_units

    def _chunk_ids(self,
                   ids: frozenset[int],
                   n_parts: int) -> List[set[int]]:

        lst = sorted(list(ids))
        base, rem = divmod(len(lst), n_parts)
        chunks, idx = [], 0
        for i in range(n_parts):
            size = base + (1 if i < rem else 0)
            chunks.append(set(lst[idx: idx + size]))
            idx += size
        return chunks

    def _create_child_samples(self,
                              pulled_samples: List[Sample],
                              length: float, n_parents: int) -> Tuple[Sample, Sample]:
        """
        Creates two new child samples from a list of parent samples.

        For geometric sampling crossover, we need to preserve inclusion probabilities.

        Key mathematical constraints:
        - Each unit's inclusion probability must be preserved exactly
        - For unit u in parent1 only: u contributes `length` to its inclusion prob from parent1
        - For unit u in parent2 only: u contributes `length` to its inclusion prob from parent2
        - For unit u in both parents: u contributes `2*length` to its inclusion prob

        To preserve inclusion probabilities:
        - Overlap units must appear in BOTH children (each contributes `length`)
        - Unique_to_1 units must go to exactly ONE child (contributes `length`)
        - Unique_to_2 units must go to exactly ONE child (contributes `length`)

        For genetic diversity while preserving probabilities:
        - Child1 gets: overlap + unique_to_1 (from parent1) + nothing from parent2's unique
        - Child2 gets: overlap + unique_to_2 (from parent2) + nothing from parent1's unique

        This way:
        - Overlap units: appear in both children, each with prob `length`, total = 2*length ✓
        - Unique_to_1: appear only in child1 with prob `length` ✓
        - Unique_to_2: appear only in child2 with prob `length` ✓
        """
        ids1 = pulled_samples[0].ids
        ids2 = pulled_samples[1].ids

        # Simple case: samples are identical
        if ids1 == ids2:
            return Sample(length, ids1), Sample(length, ids2)

        # Find overlap and unique units
        overlap = ids1 & ids2
        unique_to_1 = ids1 - ids2
        unique_to_2 = ids2 - ids1

        # Child 1: overlap + all unique_to_1 (preserves parent1's contribution)
        # Child 2: overlap + all unique_to_2 (preserves parent2's contribution)
        child1_ids = overlap | unique_to_1
        child2_ids = overlap | unique_to_2

        child1 = Sample(length, frozenset(child1_ids))
        child2 = Sample(length, frozenset(child2_ids))
        return child1, child2

    def _update_leftovers(self,
                          leftovers: List[Optional[Sample]],
                          pulled_samples: List[Sample], length: float):
        """Calculates the remaining probability for each sample and updates leftovers."""
        for i, r in enumerate(pulled_samples):
            rem = r.probability - length
            if rem > EPSILON:
                leftovers[i] = Sample(rem, r.ids, index=[-1, []])
            else:
                leftovers[i] = None

    def _classify_sample_by_partition(self,
                                      sample: Sample,
                                      border_units: set[int]) -> int:
        """
        Classify a sample based on its border units using the new logic.
        - Returns -1 if the sample contains NO border units.
        - Returns 1 if the sample's middle ID IS a border unit.
        - Returns 0 if the sample HAS border units, but its middle ID is NOT one.

        """
        if not sample.ids & border_units:
            return -1  # No border unit in sample

        # Determine the middle id in a stable order and check membership in border_units
        mid_idx = len(sample.ids) // 2
        mid_id = sorted(sample.ids)[mid_idx]
        if mid_id in border_units:
            return 1
        return 0

    def _sort_samples_by_partition(self,
                                   samples: List[Sample],
                                   border_units: set[int],
                                   reverse: bool = False) -> List[Sample]:
        """
        Sort samples by partition classification.

        Args:
            samples: List of samples to sort
            border_units: Set of border unit indices
            partitions: Dictionary mapping partition index to unit indices (unused
                        by new classifier but kept for signature)
            reverse: If False (default), sort 1 -> 0 -> -1
                     If True, sort -1 -> 0 -> 1
        """
        classified = []
        for sample in samples:
            part_idx = self._classify_sample_by_partition(sample, border_units)
            classified.append((part_idx, sample))

        if reverse:
            # Sort 0 -> -1 -> 1
            key_map = {0: 1, -1: 0, 1: -1}
            classified.sort(key=lambda x: key_map.get(x[0]), reverse=True)
        else:
            # Sort 1 -> -1 -> 0
            key_map = {1: 1, -1: 0, 0: -1}
            classified.sort(key=lambda x: key_map.get(x[0]), reverse=True)

        return [sample for _, sample in classified]

    def combine_n_parents(
            self,
            parents: List[DesignGenetic],
            border_units: Optional[set[int]] = None,
    ) -> tuple[DesignGenetic, DesignGenetic]:
        """
        Performs a crossover operation between N parents to produce two children.

        This method always converts parent heaps to lists and processes them
        sequentially.

        If partitions and border_units are provided, samples are sorted by partition
        before combining to reduce intersection conflicts with border units.
        """
        parents = [par.copy() for par in parents]
        n = len(parents)

        child1 = DesignGenetic(inclusions=None, rng=parents[0].rng)
        child2 = DesignGenetic(inclusions=None, rng=parents[1].rng)

        samples_list: List[List[Sample]] = []

        # --- UNIFIED LOGIC ---
        # 1. Convert heaps to lists, sorting *only* if partition info is given
        if  border_units is not None:
            # Sort lists based on partition logic
            for i, parent in enumerate(parents):
                samples = list(parent.heap)
                sorted_samples = self._sort_samples_by_partition(
                    samples, border_units
                )
                samples_list.append(sorted_samples)
        else:
            for parent in parents:
                samples_list.append(list(parent.heap))

        indices = [0] * n
        leftovers: List[Optional[Sample]] = [None] * n

        while any(i < len(samples_list[j]) for j, i in enumerate(indices)) or any(leftovers):
            # Step 1: Pull from lists or leftovers
            pulled_samples = []
            for i in range(n):
                if leftovers[i] is not None:
                    pulled_samples.append(leftovers[i])
                    leftovers[i] = None
                elif indices[i] < len(samples_list[i]):
                    pulled_samples.append(samples_list[i][indices[i]])
                    indices[i] += 1
                else:
                    pulled_samples = []
                    break

            if len(pulled_samples) != n:
                break

            # Step 2: Determine the common probability length
            length = min(r.probability for r in pulled_samples)
            if length <= EPSILON:
                continue

            # Step 3: Create the new child samples
            new_sample1, new_sample2 = self._create_child_samples(pulled_samples, length, n)

            new_sample1.index = [child1.step, []]
            new_sample2.index = [child2.step, []]
            child1.push(new_sample1)
            child2.push(new_sample2)

            # Step 4: Calculate and store any remaining sample portions
            self._update_leftovers(leftovers, pulled_samples, length)

            child1.step += 1
            child1.changes += 1
            child2.step += 1
            child2.changes += 1

        return child1, child2

    def design_fragmentation_n(
            self,
            parent: DesignGenetic,
            n_parts: int,
            random_pull: bool = False) -> List[DesignGenetic]:

        children: List[DesignGenetic] = [
            DesignGenetic(inclusions=None, rng=parent.rng) for _ in range(n_parts)
        ]

        while parent.heap:
            sample = parent.pull(random_pull)
            ids_chunks = self._chunk_ids(sample.ids, n_parts)

            for i, ids_part in enumerate(ids_chunks):
                if not ids_part:
                    continue
                weight_i = sample.probability
                s = Sample(weight_i, frozenset(ids_part), index=sample.index)
                children[i].push(s)
                children[i].changes += 1

        return children

    def combine_fragments_n(
            self, fragments: List[DesignGenetic], random_pull: bool = False
    ) -> DesignGenetic:
        n = len(fragments)
        leftovers: List[Optional[Sample]] = [None] * n

        child = DesignGenetic(inclusions=None, rng=fragments[0].rng)

        while any(leftovers) or any(frag.heap for frag in fragments):
            pulled: List[Sample] = []
            for i, frag in enumerate(fragments):
                if leftovers[i] is not None:
                    r = leftovers[i]
                    leftovers[i] = None
                else:
                    r = frag.pull(random_pull)
                pulled.append(r)

            length = min(r.probability for r in pulled)
            if length <= EPSILON:
                continue

            combined_ids: set[int] = set()
            for r in pulled:
                combined_ids |= r.ids

            child_sample = Sample(
                length, frozenset(combined_ids), index=[child.step, []]
            )
            child.push(child_sample)

            for i, r in enumerate(pulled):
                rem = r.probability - length
                if rem > EPSILON:
                    leftovers[i] = Sample(rem, r.ids, index=[-1, []])
                else:
                    leftovers[i] = None

            child.step += 1
            child.changes += 1

        return child