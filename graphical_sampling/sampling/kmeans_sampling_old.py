from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from . import Organizer
from ..structs import _Sample
from ..design import Design
from ..index import DensityDisparity


class KMeansSampling:
    def __init__(
        self,
        coordinate: NDArray,
        inclusion_probability: NDArray,
        *,
        n: int,
        n_zones: int | tuple[int, int],
        tolerance: int,
        split_size: float,
        zone_mode: str = "sweep",
        sort_method: str = "lexico",
        zonal_sort: str = "lexico",
        max_missed_samples: int = 10
    ) -> None:
        self.coords = self._normalize_coords(coordinate)
        self.probs = inclusion_probability
        self.n = n
        self.tolerance = tolerance
        self.split_size = split_size
        self.zone_mode = zone_mode
        self.sort_method = sort_method
        self.zonal_sort = zonal_sort
        self.n_zones = self._convert_n_zones(n_zones)
        self.max_missed_samples = max_missed_samples

        self.popu = Organizer(
            self.coords,
            self.probs,
            n_clusters=self.n,
            n_zones=n_zones,
            tolerance=self.tolerance,
            split_size=self.split_size,
            zone_mode=self.zone_mode,
            sort_method=self.sort_method,
            zonal_sort=self.zonal_sort,
        )
        self.rng = np.random.default_rng()

        self.design = self._generate_design()
        self.density = self._build_density()
        self.all_samples, self.all_samples_probs = self._get_all_samples_with_probs()
        # print(f'Lost Probs: {1 - np.sum(self.all_samples_probs)}')
        self.all_samples_probs *= 1/np.sum(self.all_samples_probs)

    def _generate_design(self) -> Design:
        design = self._build_design()
        if self._has_missed_samples(design):
            before_num = self.get_num_missed_samples(design)
            # print(f'WARNING: Design contains missed samples: {before_num}')
            # print('Move border units INSIDE zones...')
            self._move_border_units_inside_zones()
            new_design = self._build_design()
            after_num = self.get_num_missed_samples(new_design)
            if after_num < before_num:
                design = new_design
                # print(f'{before_num - after_num} of them have been solved.')
                # print(f'Current number of missed samples: {after_num}')
            else:
                pass
                # print('Failed to solve the design.')

        k = 0
        while self._has_missed_samples(design) and self.get_num_missed_samples(design) > self.max_missed_samples and k < 3:
            before_num = self.get_num_missed_samples(design)
            # print(f'\nWARNING: Design contains missed samples: {before_num}')
            # print('Move border units OUTSIDE zones...')
            self._move_border_units_outside_zones()
            new_design = self._build_design()
            after_num = self.get_num_missed_samples(new_design)
            if after_num < before_num:
                design = new_design
                # print(f'{before_num - after_num} of them have been solved.')
                # print(f'Current number of missed samples: {after_num}')
            else:
                pass
                # print('Failed to solve the design.')
            k += 1
        return design

    def _has_missed_samples(self, design) -> bool:
        for sample_obj in design:
            if len(sample_obj.indices) < self.n:
                return True
        return False

    def get_num_missed_samples(self, design) -> int:
        k = 0
        for sample_obj in design:
            if len(sample_obj.indices) < self.n:
                k += 1
                # print(f'Missed Samples with length {len(sample_obj.indices)}: {sample_obj}')
        return k

    def _move_border_units_inside_zones(self):
        bad_borders = self.popu.borders.get_bad_borders()

        for bid_zone, cls_set in bad_borders.items():
            bid, zn = bid_zone
            for ci, cls in enumerate(sorted(cls_set)):
                arr = self.popu.clusters[cls].zones[zn].units
                idxs = np.nonzero(arr[:, 0] == bid)[0]
                i = idxs[0]
                N = arr.shape[0]
                others = np.delete(np.arange(N), i)
                if ci == 0:
                    new_order = np.concatenate(([i], others))
                else:
                    new_order = np.concatenate((others, [i]))
                self.popu.clusters[cls].zones[zn].units = arr[new_order]

        self.popu.borders.update_from(self.popu.clusters)

    def _move_border_units_outside_zones(self):
        bad_borders = self.popu.borders.get_bad_borders()
        for (bid, zn), cls_set in bad_borders.items():
            # pick one class at random for this border
            cls = np.random.choice(list(cls_set))
            cluster = self.popu.clusters[cls]
            current_zone = cluster.zones[zn]

            # find all candidate “outside” zones
            possible_zones = self.popu.borders.get_possible_zones(bid, np.prod(self.n_zones))
            if not possible_zones:
                continue

            # compute current zone centroid
            current_centroid = current_zone.units[:, 1:3].mean(axis=0)
            # compute centroids of each candidate
            centroids = np.array([
                cluster.zones[z].units[:, 1:3].mean(axis=0)
                for z in possible_zones
            ])
            # pick the zone with minimal Euclidean distance
            best_idx = np.argmin(np.linalg.norm(centroids - current_centroid, axis=1))
            target_i = possible_zones[best_idx]
            target_zone = cluster.zones[target_i]

            # locate & remove the border‐unit row from current zone
            mask = current_zone.units[:, 0] == bid
            if not np.any(mask):
                continue
            b_idx = np.where(mask)[0][0]
            border_unit = current_zone.units[b_idx].copy()
            current_zone.units = np.delete(current_zone.units, b_idx, axis=0)

            # add that border unit to the target zone
            target_zone.units = np.vstack([target_zone.units, border_unit])

            # now pull back from target_zone the same total prob mass
            p_needed = border_unit[3]
            probs = target_zone.units[:, 3]
            cumsum = np.cumsum(probs)

            if cumsum[-1] < p_needed:
                raise ValueError(f"Not enough probability in zone {target_i} to exchange.")

            # if there’s an exact cumsum match, just slice
            exact = np.where(cumsum == p_needed)[0]
            if exact.size:
                cut = exact[0] + 1
                to_move = target_zone.units[:cut]
                remaining = target_zone.units[cut:]
            else:
                # otherwise, split the unit where cumsum crosses p_needed
                k = np.searchsorted(cumsum, p_needed)
                prev_sum = cumsum[k - 1] if k > 0 else 0.0
                take = p_needed - prev_sum

                unit_k = target_zone.units[k]
                sel = unit_k.copy();
                sel[3] = take
                rem = unit_k.copy();
                rem[3] -= take

                to_move = np.vstack([target_zone.units[:k], sel])
                remaining = np.vstack([rem, target_zone.units[k + 1:]])

            # update the two zones
            target_zone.units = remaining
            current_zone.units = np.vstack([current_zone.units, to_move])

        self.popu.borders.update_from(self.popu.clusters)


    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        return (coords - coords.min(axis=0)) / np.ptp(coords, axis=0).max()

    def _convert_n_zones(self, n):
        if self.zone_mode == "sweep":
            if isinstance(n, int):
                return n, n
        return n

    def sample(self, n_samples: int):
        samples = np.zeros((n_samples, self.n), dtype=int)
        for i in range(n_samples):
            random_number = self.rng.random()
            zone_index = np.searchsorted(
                np.arange(1, step=round(1 / np.prod(self.n_zones), self.tolerance)),
                random_number,
                side="right",
            )
            for j, cluster in enumerate(self.popu.clusters):
                unit_index = np.searchsorted(
                    cluster.zones[zone_index - 1].units[:, 3].cumsum()
                    + (zone_index - 1)
                    * round(1 / np.prod(self.n_zones), self.tolerance),
                    random_number,
                    side="right",
                )
                if np.prod(self.n_zones) != len(cluster.zones):
                    warning_msg = (
                        f"Warning: Cluster {j} has {len(cluster.zones)} zones, "
                        f"but expected {np.prod(self.n_zones)} based on n_zones={self.n_zones}"
                    )
                    print(warning_msg)
                samples[i, j] = cluster.zones[zone_index - 1].units[unit_index, 0]

        return samples


    def _build_design(self) -> Design:
        # 1. prep
        Z = int(np.prod(self.n_zones))
        zone_width = round(1.0 / Z, self.tolerance)
        design = Design(inclusion=None)

        # 2. collect *all* break‐points: zone boundaries + unit boundaries
        cuts = set()
        # zone boundaries at 0, zone_width, 2*zone_width, …, 1
        for k in range(Z + 1):
            cuts.add(round(k * zone_width, self.tolerance))

        # within each cluster, within each zone, the cumulative unit‐prob cuts
        for cl in self.popu.clusters:
            for zi, zone in enumerate(cl.zones):
                start = round(zi * zone_width, self.tolerance)
                cum = np.cumsum(zone.units[:, 3])
                for c in cum:
                    cuts.add(round(start + c, self.tolerance))

        # 3. sort & clip into [0,1]
        pts = sorted(c for c in cuts if 0.0 <= c <= 1.0)

        # 4. walk the intervals
        last = pts[0]
        for p in pts[1:]:
            length = round(p - last, self.tolerance)
            if length <= 0:
                last = p
                continue

            mid = last + length / 2.0

            # figure out which zone index across *all* clusters
            zone_edges = np.arange(1.0, step=zone_width)
            zone_idx = np.searchsorted(zone_edges, mid, side="right") - 1

            # for that zone, for each cluster find the unit
            ids = []
            for cl in self.popu.clusters:
                z = cl.zones[zone_idx]
                cum_u = z.units[:, 3].cumsum()
                # shift by zone start
                mapped = cum_u + zone_idx * zone_width
                u_idx = np.searchsorted(mapped, mid, side="right")
                ids.append(int(z.units[u_idx, 0]))

            design._push(_Sample(length, frozenset(ids)))
            last = p

        design.merge_identical()

        return design

    def _build_density(self) -> DensityDisparity:
        labels = np.argmax(self.popu.dbk.membership, axis=1)
        centroids = np.vstack([
            self.coords[labels == i].mean(axis=0) for i in range(self.n)
        ])
        return DensityDisparity(self.coords, self.probs, self.n, self.split_size, labels)

    def _get_all_samples_with_probs(self):
        samples = []
        samples_probs = []
        for sample_obj in self.design:
            if len(sample_obj.ids) < self.n:
                continue
            samples.append(list(sample_obj.ids))
            samples_probs.append(sample_obj.prob)
        return np.array(samples), np.array(samples_probs)

    @cached_property
    def _get_density_scores(self):
        return self.density.score(self.all_samples)

    def expected_score(self, all_samples_scores=None):
        if all_samples_scores is None:
            all_samples_scores = self._get_density_scores
        return np.sum(all_samples_scores * self.all_samples_probs)

    def var_score(self, all_samples_scores=None):
        if all_samples_scores is None:
            all_samples_scores = self._get_density_scores
        return np.sum(all_samples_scores**2 * self.all_samples_probs) - self.expected_score(all_samples_scores)**2
