import sys

sys.path.insert(0, "..")

from scipy.stats import multivariate_normal
import numpy as np

from cluster.cluster import KMeansClustering, SpectralClustering


# test 10 times
def test_k_means_gaussian_mixture():
    np.random.seed(42)

    mvn1 = multivariate_normal(mean=np.array([1, 1]), cov=np.eye(2) * 0.01)
    mvn2 = multivariate_normal(mean=np.array([0.5, 0.5]), cov=np.eye(2) * 0.02)
    mvn3 = multivariate_normal(mean=np.array([0, 1]), cov=np.eye(2) * 0.02)

    rvs1 = mvn1.rvs(8)
    rvs2 = mvn2.rvs(8)
    rvs3 = mvn3.rvs(8)

    k_means_clustering = KMeansClustering(k=3, rep=100)
    k_means_clustering.cluster(points=np.vstack([rvs1, rvs2, rvs3]))

    labels = k_means_clustering.get_labels()

    assert np.sum(labels[0:8] == labels[0]) == 8
    assert np.sum(labels[8:16] == labels[8]) == 8
    assert np.sum(labels[16:] == labels[16]) == 8


# test 10 times
def test_spectral_gaussian_mixture():
    np.random.seed(42 * 42)

    mvn1 = multivariate_normal(mean=np.array([1, 1]), cov=np.eye(2) * 0.01)
    mvn2 = multivariate_normal(mean=np.array([0.5, 0.5]), cov=np.eye(2) * 0.02)
    mvn3 = multivariate_normal(mean=np.array([0, 1]), cov=np.eye(2) * 0.02)

    rvs1 = mvn1.rvs(8)
    rvs2 = mvn2.rvs(8)
    rvs3 = mvn3.rvs(8)

    spectral_clustering = SpectralClustering(k=3, graph_param=3)
    spectral_clustering.cluster(points=np.vstack([rvs1, rvs2, rvs3]))

    labels = spectral_clustering.get_labels()

    assert np.sum(labels[0:8] == labels[0]) == 8
    assert np.sum(labels[8:16] == labels[8]) == 8
    assert np.sum(labels[16:] == labels[16]) == 8


def test_spectral_circles():
    np.random.seed(42)

    N1 = 25
    r1 = 1 / 4
    N2 = 75
    r2 = 3 / 4

    x1 = 2 * np.pi * np.random.rand(N1)
    x2 = 2 * np.pi * np.random.rand(N2)

    r1 = r1 + np.random.rand(N1) * 0.1
    r2 = r2 + np.random.rand(N2) * 0.1

    points = np.vstack(
        [
            np.concatenate([r1 * np.cos(x1), r2 * np.cos(x2)]),
            np.concatenate([r1 * np.sin(x1), r2 * np.sin(x2)]),
        ]
    ).T

    # kNN graph
    spectral_clustering = SpectralClustering(k=2, graph_param=5)
    spectral_clustering.cluster(points=points)

    labels = spectral_clustering.get_labels()

    assert np.sum(labels[:25] == labels[0]) == 25
    assert np.sum(labels[25:] == labels[25]) == 75

    # mkNN graph
    spectral_clustering = SpectralClustering(
        k=2, graph_param=10, sparse=False, graph="mkNN"
    )
    spectral_clustering.cluster(points=points)

    labels = spectral_clustering.get_labels()

    assert np.sum(labels[:25] == labels[0]) == 25
    assert np.sum(labels[25:] == labels[25]) == 75

    # eps-graph
    spectral_clustering = SpectralClustering(
        k=2, graph_param=0.4, sparse=False, graph="eps"
    )
    spectral_clustering.cluster(points=points)

    labels = spectral_clustering.get_labels()

    assert np.sum(labels[:25] == labels[0]) == 25
    assert np.sum(labels[25:] == labels[25]) == 75

    # fc-graph does not work for this problem
