import copy
import time
from functools import lru_cache
from math import isclose
from typing import List, Optional

from geometric_sampling.design import DesignGenetic
from geometric_sampling.structs import Sample

import numpy as np


class GeneticOptimizer:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def partition_design(
        self, fip_list: list[float], num_partitions: int
    ) -> tuple[dict[int, list[int]], set[int]]:
        target = sum(fip_list) / num_partitions
        partitions, border_units = {}, set()
        current_partition, cumulative_sum = 0, 0.0
        partitions[current_partition] = []

        for index, fip in enumerate(fip_list):
            if (
                isclose(cumulative_sum + fip, target, abs_tol=1e-9)
                or cumulative_sum + fip < target
            ):
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

    def _chunk_ids(self, ids: frozenset[int], n_parts: int) -> List[set[int]]:
        lst = sorted(ids)
        base, rem = divmod(len(lst), n_parts)
        chunks, idx = [], 0
        for i in range(n_parts):
            size = base + (1 if i < rem else 0)
            chunks.append(set(lst[idx : idx + size]))
            idx += size
        return chunks

    def design_fragmentation_n(
        self, parent: DesignGenetic, n_parts: int, random_pull: bool = False
    ) -> List[DesignGenetic]:
        children: List[DesignGenetic] = [
            DesignGenetic(inclusions=None, rng=parent.rng) for _ in range(n_parts)
        ]

        while parent.heap:
            sample = parent.pull(random_pull)
            ids_chunks = self._chunk_ids(sample.ids, n_parts)
            # total_ids = len(sample.ids)

            for i, ids_part in enumerate(ids_chunks):
                if not ids_part:
                    continue
                weight_i = sample.probability
                s = Sample(weight_i, frozenset(ids_part), index=sample.index)
                children[i].push(s)
                children[i].changes += 1

        return children

    @lru_cache(maxsize=None)
    def _chunk_ids_cached(self, ids_tuple: tuple[int, ...], n_parts: int) -> List[set]:
        return self._chunk_ids(frozenset(ids_tuple), n_parts)

    def combine_n_parents(
        self,
        parents: List[DesignGenetic],
        random_pull: bool = False,
    ) -> tuple[DesignGenetic, DesignGenetic]:

        parents = [par.copy() for par in parents]

        n = len(parents)
        leftovers: List[Optional[Sample]] = [None] * n
        child_design = DesignGenetic(inclusions=None, rng=parents[0].rng)
        child_design2 = DesignGenetic(inclusions=None, rng=parents[1].rng)

        # Continue while each parent has either a leftover or available heap entries
        while all(leftovers[i] is not None or parents[i].heap for i in range(n)):
            pulled: List[Sample] = []
            for i, par in enumerate(parents):
                if leftovers[i] is not None:
                    r = leftovers[i]
                    leftovers[i] = None
                else:
                    r = par.pull(random_pull)
                pulled.append(r)

            length = min(r.probability for r in pulled)
            
            all_chunks = [self._chunk_ids(r.ids, n) for r in pulled]
            child_ids: set[int] = set()
            child_ids2: set[int] = set()
            for i in range(n):
                child_ids |= all_chunks[i][i]
                child_ids2 |= all_chunks[i][(i + 1) % n]
            child = Sample(length, frozenset(child_ids), index=[child_design.step, []])
            child2 = Sample(
                length, frozenset(child_ids2), index=[child_design2.step, []]
            )

            child_design.push(child)
            child_design2.push(child2)

            for i, r in enumerate(pulled):
                rem = r.probability - length
                if rem > 1e-12:
                    leftovers[i] = Sample(rem, r.ids, index=[-1, []])
                else:
                    leftovers[i] = None

            child_design.step += 1
            child_design.changes += 1
            child_design2.step += 1
            child_design2.changes += 1

        return child_design, child_design2

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

            # idx = pulled[0].index

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
