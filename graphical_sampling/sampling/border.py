from collections import defaultdict

import numpy as np


class Borders:
    def __init__(self, border_ids: set):
        self.border_ids = border_ids
        self._reset_maps()

    def _reset_maps(self):
        self.border_to_locations = defaultdict(list)
        self.border_zone_to_clusters = defaultdict(set)

    def add(self, border_id: int, cluster_idx: int, zone_idx: int):
        self.border_to_locations[border_id].append((cluster_idx, zone_idx))
        self.border_zone_to_clusters[(border_id, zone_idx)].add(cluster_idx)

    def build_from(self, clusters):
        """Scan your clusters/zones once and populate the maps."""
        self._reset_maps()
        for ci, cluster in enumerate(clusters):
            for zi, zone in enumerate(cluster.zones):
                zone_ids = set(zone.units[:, 0])
                common_ids = self.border_ids.intersection(zone_ids)
                for bid in common_ids:
                    self.add(bid, ci, zi)

    def update_from(self, clusters):
        self.build_from(clusters)

    def get_bad_borders(self):
        return {
            key: clusters
            for key, clusters in self.border_zone_to_clusters.items()
            if len(clusters) > 1
        }

    def get_possible_zones(self, border_id: int, n):
        all_zones = set(np.arange(n))
        occupied_zones = set(np.array(self.border_to_locations[border_id])[:, 1])
        return list(all_zones - occupied_zones)
