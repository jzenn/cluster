import sys

sys.path.insert(0, "..")

import pytest
import numpy as np

from cluster.utils import (
    get_similarity_matrix,
    get_random_similarity_matrix,
    get_distance_matrix_from_similarity_matrix,
    get_similarity_matrix_from_distance_matrix,
)


# test 10 times
@pytest.mark.parametrize("_", range(10))
@pytest.mark.parametrize("d", range(3, 7))
def test_similarity_matrix_distance_itself(_, d):
    points = np.random.rand(10, d)

    S = get_similarity_matrix(points)

    assert np.diag(S).sum() == 10
    assert np.min(S) < 1.0


# test 10 times
@pytest.mark.parametrize("_", range(10))
@pytest.mark.parametrize("d", range(3, 7))
def test_similarity_matrix_symmetric(_, d):
    points = np.random.rand(10, d)

    S = get_similarity_matrix(points)

    assert np.sum(S == S.T) == 10 ** 2


# test 10 times
@pytest.mark.parametrize("_", range(10))
@pytest.mark.parametrize("d", range(3, 7))
def test_similarity_matrix_vectorized_iterative(_, d):
    points = np.random.rand(10, d)

    S1 = get_similarity_matrix(points, vectorized=True)
    S2 = get_similarity_matrix(points, vectorized=False)

    np.testing.assert_allclose(S1, S2)


# test 10 times
@pytest.mark.parametrize("_", range(10))
@pytest.mark.parametrize("d", range(3, 7))
def test_similarity_matrix_random(_, d):
    np.random.seed(42)
    points = np.random.rand(10, d)
    S1 = get_similarity_matrix(points)

    np.random.seed(42)
    S2 = get_random_similarity_matrix(n=10, d=d)

    np.testing.assert_allclose(S1, S2)


# test 10 times
@pytest.mark.parametrize("_", range(10))
def test_get_distance_matrix_from_similarity_matrix(_):
    points = np.random.rand(10, 5)

    D = get_similarity_matrix(points, distance=True)
    S = get_similarity_matrix(points)

    np.testing.assert_allclose(D, get_distance_matrix_from_similarity_matrix(S))


# test 10 times
@pytest.mark.parametrize("_", range(10))
def test_get_similarity_matrix_from_distance_matrix(_):
    points = np.random.rand(10, 5)

    D = get_similarity_matrix(points, distance=True)
    S = get_similarity_matrix(points)

    np.testing.assert_allclose(get_similarity_matrix_from_distance_matrix(D), S)
