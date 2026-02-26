from typing import Iterator
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt

from .population import Population
from .structs import MaxHeap, Sample
from .order import Order  # <-- Import the new Order class


class Design:
    def __init__(
            self,
            population: Population,
            num_zones: int = 1,
            order: Order | None = None,
    ):
        self.pop = population
        self.num_zones = num_zones
        self.order = order if order is not None else Order.from_indices(population)
        self.heaps = [MaxHeap() for _ in range(num_zones)]
        self.rng = np.random.default_rng()

        self._build()

    def copy(self) -> Design:
        new_design = Design.__new__(Design)

        new_design.pop = self.pop
        new_design.num_zones = self.num_zones
        new_design.order = self.order
        new_design.rng = np.random.default_rng()
        new_design.heaps = [h.copy() for h in self.heaps]

        return new_design

    def _build(self):
        events: list[tuple[float, str, int]] = []
        level: float = 0

        for i in self.order.get():
            p = self.pop.inclusions[i]
            next_level = level + p
            if next_level < 1 - 1e-9:
                events.append((level, "start", i))
                events.append((next_level, "end", i))
                level = next_level
            elif next_level > 1 + 1e-9:
                events.append((level, "start", i))
                events.append((1, "end", i))
                events.append((0, "start", i))
                events.append((next_level - 1, "end", i))
                level = next_level - 1
            else:
                events.append((level, "start", i))
                events.append((1, "end", i))
                level = 0

        for i in range(self.num_zones + 1):
            events.append((i / self.num_zones, "boundary", -1))

        events.sort()
        active = set()
        last_point: float = 0

        for point, event_type, bar_index in events:
            if point > last_point + 1e-9:
                if active:
                    midpoint = (last_point + point) / 2
                    zone_idx = min(int(midpoint * self.num_zones), self.num_zones - 1)

                    length = round(point - last_point, 9)
                    self._push(zone_idx, Sample(length, frozenset(active)))

            if event_type == "start":
                active.add(bar_index)
            elif event_type == "end":
                active.remove(bar_index)

            last_point = point

    def exchange(self, zones: int = 1, random_pull: bool = False, switch_coef: float = 0.5) -> None:
        if not (0 < zones <= self.num_zones):
            raise ValueError(
                f"zones_count must be greater than 0 and less than or equal to {self.num_zones}. Got {zones}."
            )

        valid_zones = [i for i, h in enumerate(self.heaps) if len(h) >= 2]
        if not valid_zones:
            return

        actual_count = min(zones, len(valid_zones))
        selected_zones = self.rng.choice(valid_zones, size=actual_count, replace=False)

        for zone_idx in selected_zones:
            zone_idx = int(zone_idx)

            r1 = self._pull(zone_idx, random_pull)
            r2 = self._pull(zone_idx, random_pull)

            if r1.ids == r2.ids:
                self._push(zone_idx, Sample(r1.prob + r2.prob, r1.ids))
            else:
                # _switch can now return 2 or 4 items; _push elegantly handles both
                self._push(zone_idx, *self._switch(r1, r2, switch_coef))

    def _pull(self, zone_idx: int, random: bool = False) -> Sample:
        if random:
            return self.heaps[zone_idx].random_pop()
        return self.heaps[zone_idx].pop()

    def _push(self, zone_idx: int, *args: Sample) -> None:
        for r in args:
            if not r.almost_zero():
                self.heaps[zone_idx].push(r)

    def _switch(self, r1: Sample, r2: Sample, coef: float = 0.5) -> tuple[Sample, ...]:
        diff1 = list((r1.ids - r2.ids) - self.order.fixed_ids)
        diff2 = list((r2.ids - r1.ids) - self.order.fixed_ids)

        if not diff1 or not diff2:
            return r1, r2

        length = coef * min(r1.prob, r2.prob)
        n1 = self.rng.choice(diff1)
        n2 = self.rng.choice(diff2)

        return (
            Sample(length, r1.ids - {n1} | {n2}),
            Sample(r1.prob - length, r1.ids),
            Sample(length, r2.ids - {n2} | {n1}),
            Sample(r2.prob - length, r2.ids),
        )

    def merge_identical(self):
        for i in range(self.num_zones):
            dic = {}
            for r in self.heaps[i]:
                dic.setdefault(r.ids, 0.0)
                dic[r.ids] += r.prob
            self.heaps[i] = MaxHeap(
                initial_heap=[Sample(length, ids) for ids, length in dic.items()]
            )

    def show(self, save: bool = False) -> None:
        initial_level: float = 0
        cmap_names = ['Blues', 'Greens', 'Wistia', 'Oranges', 'Reds', 'Purples']
        plt.figure(figsize=(8, 4.5))

        for zone_idx, heap in enumerate(self.heaps):
            cmap_name = cmap_names[zone_idx % len(cmap_names)]
            cmap = plt.get_cmap(cmap_name)

            samples = sorted(list(heap), key=lambda s: s.prob, reverse=True)
            num_samples = len(samples)

            for sample_idx, r in enumerate(samples):
                if num_samples > 1:
                    intensity = 0.9 - (0.6 * (sample_idx / (num_samples - 1)))
                else:
                    intensity = 0.7

                sample_color = cmap(intensity)

                for i in r.ids:
                    plt.plot(
                        [i, i],
                        [initial_level, initial_level + r.prob],
                        color=sample_color,
                        linewidth=1,
                        solid_capstyle='butt'
                    )
                initial_level += r.prob

        for i in range(1, self.num_zones):
            plt.axhline(y=i / self.num_zones, color='gray', linestyle='--', alpha=0.4)

        plt.ylabel("Probability")
        plt.xlabel("Population Units")
        plt.title("Design")
        if save:
            plt.savefig("design.png", dpi=300)
        plt.show()

    def __iter__(self) -> Iterator[Sample]:
        return chain.from_iterable(self.heaps)

    def __len__(self) -> int:
        return sum(len(h) for h in self.heaps)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Design):
            return NotImplemented
        return self.heaps == other.heaps

    def __hash__(self) -> int:
        return hash(tuple(self.heaps))
