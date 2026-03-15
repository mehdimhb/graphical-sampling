import numpy as np
import graphical_sampling as gs


def test_population():
    rng = gs.random.rng(42)
    tolerance = 5
    total_zone_count = 0
    sum_of_each_zone = 0
    all_zones_units_positive = 0
    all_clusters_units_positive = 0

    n_iterations = 30

    for _ in range(n_iterations):
        N = int(np.random.choice([100, 150, 200]))
        n = int(np.random.choice([4, 5, 6, 7, 8]))
        num_zone_in_d = int(np.random.choice([2, 3, 4]))
        coordinates = rng.random_coordinates((N, 2))
        probabilities = rng.unequal_probabilities(n, N)

        population = gs.sampling.Organizer(
            coordinates,
            probabilities,
            n_clusters=n,
            n_zones=(num_zone_in_d, num_zone_in_d),
            tolerance=tolerance,
        )

        total_zones = 0
        sum_of_zone_successes = 0
        all_zones_units_positive_successes = 0
        all_clusters_units_positive_successes = 0

        for cluster in population.clusters:
            for zone in cluster.zones:
                total_zones += 1
                if np.all(
                    np.abs(1 / num_zone_in_d**2 - np.sum(zone.units[:, 3]))
                    < 10**-tolerance
                ):
                    sum_of_zone_successes += 1
                all_zones_units_positive_successes += int(np.all(zone.units[:, 3] >= 0))
            all_clusters_units_positive_successes += int(
                np.all(cluster.units[:, 3] >= 0)
            )

        if total_zones == num_zone_in_d * num_zone_in_d * n:
            total_zone_count += 1
        if sum_of_zone_successes == num_zone_in_d * num_zone_in_d * n:
            sum_of_each_zone += 1
        if all_zones_units_positive_successes == num_zone_in_d * num_zone_in_d * n:
            all_zones_units_positive += 1
        if all_clusters_units_positive_successes == n:
            all_clusters_units_positive += 1

    assert sum_of_each_zone == n_iterations
    assert total_zone_count == n_iterations
    assert all_zones_units_positive == n_iterations
    assert all_clusters_units_positive == n_iterations
