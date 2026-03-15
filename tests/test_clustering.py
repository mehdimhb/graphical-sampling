import numpy as np
import graphical_sampling as gs


def test_soft_balanced_kmeans():
    rng = gs.random.rng(42)
    np_rng = np.random.default_rng(42)
    tolerance = 5

    each_cluster_sum_to_one = 0
    sum_of_clusters_equal_n = 0
    all_probs_positive = 0

    n_iterations = 50

    for _ in range(n_iterations):
        N = int(np_rng.choice([100, 150, 200]))
        n = int(np_rng.choice([3, 4, 5, 6]))
        coordinates = rng.random_coordinates((N, 2))
        probabilities = rng.unequal_probabilities(n, N)
        kmeans = gs.clustering.SoftBalancedKMeans(n, tolerance=tolerance)
        kmeans.fit(coordinates, probabilities)

        each_cluster_sum_to_one += int(
            np.all(np.abs(kmeans.clusters_sum - 1) < 10**-tolerance)
        )
        sum_of_clusters_equal_n += int(
            np.all((np.sum(kmeans.clusters_sum) - n) < 10**-tolerance)
        )
        all_probs_positive += int(np.all(kmeans.fractional_labels >= 0))

    assert each_cluster_sum_to_one == n_iterations
    assert sum_of_clusters_equal_n == n_iterations
    assert all_probs_positive == n_iterations
