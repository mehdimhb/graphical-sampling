from functools import lru_cache
from math import isclose
from typing import List, Optional, Tuple

from geometric_sampling.design import DesignGenetic
from geometric_sampling.structs import Sample

import numpy as np


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
                    current_partition += 1
                    partitions[current_partition] = []
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

    def _pull_from_sources(self,
                           sources: List[DesignGenetic],
                           leftovers: List[Optional[Sample]],
                           random_pull: bool) -> List[Sample]:
        """Pulls the next sample for each parent from either its heap or leftovers."""
        pulled = []
        for i, source in enumerate(sources):
            if leftovers[i] is not None:
                sample = leftovers[i]
                leftovers[i] = None
            else:
                sample = source.pull(random_pull)
            pulled.append(sample)
        return pulled

    def _create_child_samples(self,
                              pulled_samples: List[Sample],
                              length: float, n_parents: int) -> Tuple[Sample, Sample]:
        """Creates two new child samples from a list of parent samples."""
        all_chunks = [self._chunk_ids(r.ids, n_parents) for r in pulled_samples]

        child1_ids: set[int] = set()
        child2_ids: set[int] = set()

        for i in range(n_parents):
            child1_ids.update(all_chunks[i][i])
            child2_ids.update(all_chunks[i][(i + 1) % n_parents])

        child1 = Sample(length, frozenset(child1_ids))
        child2 = Sample(length, frozenset(child2_ids))
        return child1, child2

    def _update_leftovers(self,
                          leftovers: List[Optional[Sample]],
                          pulled_samples: List[Sample], length: float):
        """Calculates the remaining probability for each sample and updates leftovers."""
        for i, r in enumerate(pulled_samples):
            rem = r.probability - length
            if rem > 1e-12:
                leftovers[i] = Sample(rem, r.ids, index=[-1, []])
            else:
                leftovers[i] = None

    def combine_n_parents(
            self,
            parents: List[DesignGenetic],
            random_pull: bool = False,
    ) -> tuple[DesignGenetic, DesignGenetic]:
        """
        Performs a crossover operation between N parents to produce two children.
        This method is now a high-level coordinator for the crossover process.
        """
        parents = [par.copy() for par in parents]
        n = len(parents)

        child1 = DesignGenetic(inclusions=None, rng=parents[0].rng)
        child2 = DesignGenetic(inclusions=None, rng=parents[1].rng)
        leftovers: List[Optional[Sample]] = [None] * n

        while any(leftovers) or all(p.heap for p in parents):
            # Step 1: Get the next sample from each parent source
            pulled_samples = self._pull_from_sources(parents, leftovers, random_pull)

            # Step 2: Determine the common probability length
            length = min(r.probability for r in pulled_samples)
            if length <= 1e-12:
                continue

            # Step 3: Create the new child samples by chunking and recombining
            new_sample1, new_sample2 = self._create_child_samples(pulled_samples, length, n)

            # Assign indices and push to children
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
            if length <= 0:
                break

            combined_ids: set[int] = set()
            for r in pulled:
                combined_ids |= r.ids

            child_sample = Sample(
                length, frozenset(combined_ids), index=[child.step, []]
            )
            child.push(child_sample)

            for i, r in enumerate(pulled):
                rem = r.probability - length
                if rem > 0:
                    leftovers[i] = Sample(rem, r.ids, index=[-1, []])
                else:
                    leftovers[i] = None

            child.step += 1
            child.changes += 1

        return child
