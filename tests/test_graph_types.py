import sys

sys.path.insert(0, "..")

import numpy as np

from cluster.utils import (
    eps_graph_from_similarity_matrix,
    fc_graph_from_similarity_matrix,
    kNN_graph_from_similarity_matrix,
    mkNN_graph_from_similarity_matrix,
    get_similarity_matrix_from_distance_matrix,
)


def test_fc_graph_from_similarity_matrix():
    similarity_matrix = np.array(
        [
            [1.0, 0.5, 0.0, 0.2],
            [0.5, 1.0, 0.3, 0.2],
            [0.0, 0.3, 1.0, 0.0],
            [0.2, 0.2, 0.0, 1.0],
        ]
    )
    G = fc_graph_from_similarity_matrix(similarity_matrix)

    node_list = G.get_node_list()
    for node in node_list:
        assert node.get_degree() == 3

    for node in node_list:
        neighbors = G.get_neighbors(node)

        # all nodes need to be neighbors to the current node
        for n in set(node_list) - set([node]):
            assert n in neighbors

        # the current node itself must not be part of its own neighbors
        assert node not in neighbors


def test_eps_graph_from_distance_matrix():
    distance_matrix = np.array(
        [
            [0.0, 0.5, 1.0, 0.2],
            [0.5, 0.0, 0.3, 0.2],
            [1.0, 0.3, 0.0, 1.0],
            [0.2, 0.2, 1.0, 0.0],
        ]
    )

    # test for eps = 0.25
    G = eps_graph_from_similarity_matrix(distance_matrix, eps=0.25, distance=True)
    n1, n2, n3, n4 = G.get_node_list()

    neighbors = G.get_neighbors(n1)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 not in neighbors
    assert n4 in neighbors

    neighbors = G.get_neighbors(n2)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 not in neighbors
    assert n4 in neighbors

    neighbors = G.get_neighbors(n3)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 not in neighbors
    assert n4 not in neighbors

    neighbors = G.get_neighbors(n4)
    assert n1 in neighbors
    assert n2 in neighbors
    assert n3 not in neighbors
    assert n4 not in neighbors

    # test for eps = 0.35
    G = eps_graph_from_similarity_matrix(distance_matrix, eps=0.35, distance=True)
    n1, n2, n3, n4 = G.get_node_list()

    neighbors = G.get_neighbors(n1)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 not in neighbors
    assert n4 in neighbors

    neighbors = G.get_neighbors(n2)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 in neighbors
    assert n4 in neighbors

    neighbors = G.get_neighbors(n3)
    assert n1 not in neighbors
    assert n2 in neighbors
    assert n3 not in neighbors
    assert n4 not in neighbors

    neighbors = G.get_neighbors(n4)
    assert n1 in neighbors
    assert n2 in neighbors
    assert n3 not in neighbors
    assert n4 not in neighbors


def test_kNN_graph_from_similarity_matrix():
    distance_matrix = np.array(
        [
            [0.0, 0.5, 1.0, 0.2],
            [0.5, 0.0, 0.3, 0.2],
            [1.0, 0.3, 0.0, 0.9],
            [0.2, 0.2, 0.9, 0.0],
        ]
    )

    similarity_matrix = get_similarity_matrix_from_distance_matrix(distance_matrix)

    # k = 2
    G = kNN_graph_from_similarity_matrix(similarity_matrix, k=2)
    n1, n2, n3, n4 = G.get_node_list()

    neighbors = G.get_neighbors(n1)
    assert n1 not in neighbors
    assert n2 in neighbors
    assert n3 not in neighbors
    assert n4 in neighbors

    neighbors = G.get_neighbors(n2)
    assert n1 in neighbors
    assert n2 not in neighbors
    assert n3 in neighbors
    assert n4 in neighbors

    neighbors = G.get_neighbors(n3)
    assert n1 not in neighbors
    assert n2 in neighbors
    assert n3 not in neighbors
    assert n4 in neighbors

    neighbors = G.get_neighbors(n4)
    assert n1 in neighbors
    assert n2 in neighbors
    assert n3 in neighbors
    assert n4 not in neighbors

    # k = 1
    G = kNN_graph_from_similarity_matrix(similarity_matrix, k=1)
    n1, n2, n3, n4 = G.get_node_list()

    neighbors = G.get_neighbors(n1)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 not in neighbors
    assert n4 in neighbors

    neighbors = G.get_neighbors(n2)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 in neighbors
    assert n4 in neighbors

    neighbors = G.get_neighbors(n3)
    assert n1 not in neighbors
    assert n2 in neighbors
    assert n3 not in neighbors
    assert n4 not in neighbors

    neighbors = G.get_neighbors(n4)
    assert n1 in neighbors
    assert n2 in neighbors
    assert n3 not in neighbors
    assert n4 not in neighbors


def test_mkNN_graph_from_similarity_matrix():
    distance_matrix = np.array(
        [
            [0.0, 0.5, 1.0, 0.25],
            [0.5, 0.0, 0.3, 0.2],
            [1.0, 0.3, 0.0, 0.9],
            [0.25, 0.2, 0.9, 0.0],
        ]
    )

    similarity_matrix = get_similarity_matrix_from_distance_matrix(distance_matrix)

    # k = 2
    G = mkNN_graph_from_similarity_matrix(similarity_matrix, k=2)
    n1, n2, n3, n4 = G.get_node_list()

    neighbors = G.get_neighbors(n1)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 not in neighbors
    assert n4 in neighbors

    neighbors = G.get_neighbors(n2)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 in neighbors
    assert n4 in neighbors

    neighbors = G.get_neighbors(n3)
    assert n1 not in neighbors
    assert n2 in neighbors
    assert n3 not in neighbors
    assert n4 not in neighbors

    neighbors = G.get_neighbors(n4)
    assert n1 in neighbors
    assert n2 in neighbors
    assert n3 not in neighbors
    assert n4 not in neighbors

    # k = 1
    G = mkNN_graph_from_similarity_matrix(similarity_matrix, k=1)
    n1, n2, n3, n4 = G.get_node_list()

    neighbors = G.get_neighbors(n1)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 not in neighbors
    assert n4 not in neighbors

    neighbors = G.get_neighbors(n2)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 not in neighbors
    assert n4 in neighbors

    neighbors = G.get_neighbors(n3)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 not in neighbors
    assert n4 not in neighbors

    neighbors = G.get_neighbors(n4)
    assert n1 not in neighbors
    assert n2 in neighbors
    assert n3 not in neighbors
    assert n4 not in neighbors
