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
                              length: float, n_parents: int) -> List[Optional[Sample]]:
        """
        Creates new child samples using Chunking logic.

        Returns a list of Optional[Sample].
        If a generated child has the wrong size (due to ID conflict/vanishing),
        it returns None for that specific child slot.
        """
        # We expect all children to have the same size as the first parent
        expected_size = len(pulled_samples[0].ids)

        all_chunks = [self._chunk_ids(r.ids, n_parents) for r in pulled_samples]

        valid_children: List[Optional[Sample]] = []

        for i in range(n_parents):
            # Construct candidate IDs for Child i using Round Robin chunk selection
            # Child i takes chunk i from Parent 0, chunk i+1 from Parent 1, etc.
            candidate_ids: set[int] = set()

            for parent_idx in range(n_parents):
                # Ensure we take distinct chunks from distinct parents
                chunk_index = (i + parent_idx) % n_parents
                chunk_to_take = all_chunks[parent_idx][chunk_index]
                candidate_ids.update(chunk_to_take)

            # QUALITY CONTROL:
            # Survival of the fittest: if the size is wrong, the sample dies (becomes None).
            if len(candidate_ids) == expected_size:
                valid_children.append(Sample(length, frozenset(candidate_ids)))
            else:
                valid_children.append(None)

        return valid_children

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
        """Classify a sample based on its border units."""
        if not sample.ids & border_units:
            return -1  # No border unit in sample

        # Determine the middle id in a stable order
        mid_idx = len(sample.ids) // 2
        mid_id = sorted(sample.ids)[mid_idx]
        if mid_id in border_units:
            return 1
        return 0

    def _sort_samples_by_partition(self,
                                   samples: List[Sample],
                                   border_units: set[int],
                                   reverse: bool = False) -> List[Sample]:
        """Sort samples by partition classification."""
        classified = []
        for sample in samples:
            part_idx = self._classify_sample_by_partition(sample, border_units)
            classified.append((part_idx, sample))

        if reverse:
            key_map = {0: 1, -1: 0, 1: -1}
            classified.sort(key=lambda x: key_map.get(x[0]), reverse=True)
        else:
            key_map = {1: 1, -1: 0, 0: -1}
            classified.sort(key=lambda x: key_map.get(x[0]), reverse=True)

        return [sample for _, sample in classified]

    def combine_n_parents(
            self,
            parents: List[DesignGenetic],
            border_units: Optional[set[int]] = None,
    ) -> tuple[DesignGenetic, DesignGenetic]:
        """
        Performs a crossover operation between N parents to produce children.
        Discards 'defective' children (where samples vanished due to conflict).
        """
        parents = [par.copy() for par in parents]
        n = len(parents)

        # Initialize N child designs
        children_designs = [DesignGenetic(inclusions=None, rng=parents[0].rng) for _ in range(n)]

        samples_list: List[List[Sample]] = []

        if border_units is not None:
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

            # Step 3: Create the new child samples (returns list of Optional[Sample])
            generated_samples = self._create_child_samples(pulled_samples, length, n)

            # Step 3.5: Distribute valid samples to valid designs
            for i, sample in enumerate(generated_samples):
                if sample is not None:
                    # --- HEALTHY CHILD ---
                    # The sample is valid (correct size). We accept it.
                    sample.index = [children_designs[i].step, []]
                    children_designs[i].push(sample)
                    children_designs[i].step += 1
                    children_designs[i].changes += 1
                else:
                    # --- DEFECTIVE CHILD ---
                    # The sample had the wrong size (IDs vanished). We discard it.
                    pass

            # Step 4: Calculate and store any remaining sample portions
            self._update_leftovers(leftovers, pulled_samples, length)

        # Return the first two children (assuming N=2 crossover)
        return children_designs[0], children_designs[1]

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