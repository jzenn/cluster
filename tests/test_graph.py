import sys

sys.path.insert(0, "..")

import pytest
import numpy as np

from cluster.graph import Graph, Node, Edge


def test_node():
    n = Node(1.0)
    assert n.get_value() == 1.0

    n.set_attribute("test_attribute", 42)
    assert n.get_attribute("test_attribute") == 42


def test_directed_edge():
    n1 = Node(1.0)
    n2 = Node(0.0)

    e = Edge(connection=[n1, n2], directed=True, weight=1.0)

    assert e.get_from_node() == n1
    assert e.get_to_node() == n2

    assert n1.get_attribute("in_degree") == 0
    assert n1.get_attribute("out_degree") == 1

    assert n2.get_attribute("in_degree") == 1
    assert n2.get_attribute("out_degree") == 0

    assert n1.get_degree() == 1
    assert n2.get_degree() == 1


def test_undirected_edge():
    n1 = Node(1.0)
    n2 = Node(0.0)

    e = Edge(connection=[n1, n2], directed=False, weight=1.0)

    assert e.get_from_node() == n1
    assert e.get_to_node() == n2

    assert n1.get_attribute("in_degree") is None
    assert n1.get_attribute("out_degree") is None

    assert n2.get_attribute("in_degree") is None
    assert n2.get_attribute("out_degree") is None

    assert n1.get_degree() == 1
    assert n2.get_degree() == 1


def test_directed_graph_raise_error_for_undirected_edge():
    n1 = Node(1.0)
    n2 = Node(2.0)

    e1 = Edge([n1, n2], directed=False)

    with pytest.raises(RuntimeError):
        Graph(nodes=[n1, n2], edges=[e1], directed=True)


def test_undirected_graph_raise_error_for_directed_edge():
    n1 = Node(1.0)
    n2 = Node(2.0)

    e1 = Edge([n1, n2], directed=True)

    with pytest.raises(RuntimeError):
        Graph(nodes=[n1, n2], edges=[e1], directed=False)


def test_node_indices():
    #            e1
    # graph:  n1 -> n2
    #      e2  |  /
    #          v /   (e4 (n2->n3))
    #         n3 -> n4
    #            e3

    n1 = Node(1.0)
    n2 = Node(2.0)
    n3 = Node(3.0)
    n4 = Node(4.0)

    e1 = Edge([n1, n2])
    e2 = Edge([n1, n3])
    e3 = Edge([n3, n4])
    e4 = Edge([n2, n3])

    g1 = Graph([n1, n2, n3, n4], [e1, e2, e3, e4])

    assert g1.next_node_index == 4

    n1, n2, n3, n4 = g1.get_node_list()
    assert n1.get_attribute("node_index") == 0
    assert n2.get_attribute("node_index") == 1
    assert n3.get_attribute("node_index") == 2
    assert n4.get_attribute("node_index") == 3


def test_simple_directed_graph_connections():
    #            e1
    # graph:  n1 -> n2
    #      e2  |  /
    #          v /   (e4 (n2->n3))
    #         n3 -> n4
    #            e3

    n1 = Node(1.0)
    n2 = Node(2.0)
    n3 = Node(3.0)
    n4 = Node(4.0)

    e1 = Edge([n1, n2], directed=True)
    e2 = Edge([n1, n3], directed=True)
    e3 = Edge([n3, n4], directed=True)
    e4 = Edge([n2, n3], directed=True)

    assert n1.get_attribute("in_degree") == 0
    assert n1.get_attribute("out_degree") == 2
    assert n1.get_attribute("degree") == 2

    assert n2.get_attribute("in_degree") == 1
    assert n2.get_attribute("out_degree") == 1
    assert n2.get_attribute("degree") == 2

    assert n3.get_attribute("in_degree") == 2
    assert n3.get_attribute("out_degree") == 1
    assert n3.get_attribute("degree") == 3

    assert n4.get_attribute("in_degree") == 1
    assert n4.get_attribute("out_degree") == 0
    assert n4.get_attribute("degree") == 1

    g1 = Graph([n1, n2, n3, n4], [e1, e2, e3, e4])
    n1, n2, n3, n4 = g1.get_node_list()

    assert n1.get_attribute("in_degree") == 0
    assert n1.get_attribute("out_degree") == 2
    assert n1.get_attribute("degree") == 2

    assert n2.get_attribute("in_degree") == 1
    assert n2.get_attribute("out_degree") == 1
    assert n2.get_attribute("degree") == 2

    assert n3.get_attribute("in_degree") == 2
    assert n3.get_attribute("out_degree") == 1
    assert n3.get_attribute("degree") == 3

    assert n4.get_attribute("in_degree") == 1
    assert n4.get_attribute("out_degree") == 0
    assert n4.get_attribute("degree") == 1


def test_simple_directed_graph_neighbors():
    #            e1
    # graph:  n1 -> n2
    #      e2  |  /
    #          v /   (e4 (n2->n3))
    #         n3 -> n4
    #            e3

    n1 = Node(1.0)
    n2 = Node(2.0)
    n3 = Node(3.0)
    n4 = Node(4.0)

    e1 = Edge([n1, n2])
    e2 = Edge([n1, n3])
    e3 = Edge([n3, n4])
    e4 = Edge([n2, n3])

    g1 = Graph([n1, n2, n3, n4], [e1, e2, e3, e4])

    neighbors = g1.get_neighbors(n1)
    assert n1 not in neighbors
    assert n2 in neighbors
    assert n3 in neighbors
    assert n4 not in neighbors

    neighbors = g1.get_neighbors(n2)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 in neighbors
    assert n4 not in neighbors

    neighbors = g1.get_neighbors(n3)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 not in neighbors
    assert n4 in neighbors

    neighbors = g1.get_neighbors(n4)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 not in neighbors
    assert n4 not in neighbors


def test_simple_undirected_graph_connections():
    #            e1
    # graph:  n1 -- n2
    #      e2  |  /
    #          | /
    #         n3 -- n4
    #            e3

    n1 = Node(1.0)
    n2 = Node(2.0)
    n3 = Node(3.0)
    n4 = Node(4.0)

    e1 = Edge([n1, n2], directed=False)
    e2 = Edge([n1, n3], directed=False)
    e3 = Edge([n3, n4], directed=False)
    e4 = Edge([n2, n3], directed=False)

    assert n1.get_attribute("in_degree") is None
    assert n1.get_attribute("out_degree") is None
    assert n1.get_attribute("degree") == 2

    assert n2.get_attribute("in_degree") is None
    assert n2.get_attribute("out_degree") is None
    assert n2.get_attribute("degree") == 2

    assert n3.get_attribute("in_degree") is None
    assert n3.get_attribute("out_degree") is None
    assert n3.get_attribute("degree") == 3

    assert n4.get_attribute("in_degree") is None
    assert n4.get_attribute("out_degree") is None
    assert n4.get_attribute("degree") == 1

    g1 = Graph([n1, n2, n3, n4], [e1, e2, e3, e4], directed=False)
    n1, n2, n3, n4 = g1.get_node_list()

    assert n1.get_attribute("in_degree") is None
    assert n1.get_attribute("out_degree") is None
    assert n1.get_attribute("degree") == 2

    assert n2.get_attribute("in_degree") is None
    assert n2.get_attribute("out_degree") is None
    assert n2.get_attribute("degree") == 2

    assert n3.get_attribute("in_degree") is None
    assert n3.get_attribute("out_degree") is None
    assert n3.get_attribute("degree") == 3

    assert n4.get_attribute("in_degree") is None
    assert n4.get_attribute("out_degree") is None
    assert n4.get_attribute("degree") == 1


def test_simple_undirected_graph_neighbors():
    #            e1
    # graph:  n1 -- n2
    #      e2  |  /
    #          | /
    #         n3 -- n4
    #            e3

    n1 = Node(1.0)
    n2 = Node(2.0)
    n3 = Node(3.0)
    n4 = Node(4.0)

    e1 = Edge([n1, n2], directed=False)
    e2 = Edge([n1, n3], directed=False)
    e3 = Edge([n3, n4], directed=False)
    e4 = Edge([n2, n3], directed=False)

    g1 = Graph([n1, n2, n3, n4], [e1, e2, e3, e4], directed=False)

    neighbors = g1.get_neighbors(n1)
    assert n1 not in neighbors
    assert n2 in neighbors
    assert n3 in neighbors
    assert n4 not in neighbors

    neighbors = g1.get_neighbors(n2)
    assert n1 in neighbors
    assert n2 not in neighbors
    assert n3 in neighbors
    assert n4 not in neighbors

    neighbors = g1.get_neighbors(n3)
    assert n1 in neighbors
    assert n2 in neighbors
    assert n3 not in neighbors
    assert n4 in neighbors

    neighbors = g1.get_neighbors(n4)
    assert n1 not in neighbors
    assert n2 not in neighbors
    assert n3 in neighbors
    assert n4 not in neighbors


def test_adjacency_directed_graph():
    #            e1
    # graph:  n1 -> n2
    #      e2  |  /
    #          v /   (e4 (n2->n3))
    #         n3 -> n4
    #            e3

    n1 = Node(1.0)
    n2 = Node(2.0)
    n3 = Node(3.0)
    n4 = Node(4.0)

    e1 = Edge([n1, n2], weight=4.0)
    e2 = Edge([n1, n3], weight=5.0)
    e3 = Edge([n3, n4], weight=6.0)
    e4 = Edge([n2, n3], weight=7.0)

    g1 = Graph([n1, n2, n3, n4], [e1, e2, e3, e4])
    W1 = g1.get_W()

    adjacency_matrix = np.array(
        [
            [0.0, 4.0, 5.0, 0.0],
            [0.0, 0.0, 7.0, 0.0],
            [0.0, 0.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(W1, adjacency_matrix)


def test_adjacency_undirected_graph():
    #            e1
    # graph:  n1 -- n2
    #      e2  |  /
    #          | /
    #         n3 -- n4
    #            e3

    n1 = Node(1.0)
    n2 = Node(2.0)
    n3 = Node(3.0)
    n4 = Node(4.0)

    e1 = Edge([n1, n2], weight=4.0, directed=False)
    e2 = Edge([n1, n3], weight=5.0, directed=False)
    e3 = Edge([n3, n4], weight=6.0, directed=False)
    e4 = Edge([n2, n3], weight=7.0, directed=False)

    g1 = Graph([n1, n2, n3, n4], [e1, e2, e3, e4], directed=False)
    W1 = g1.get_W()

    adjacency_matrix = np.array(
        [
            [0.0, 4.0, 5.0, 0.0],
            [4.0, 0.0, 7.0, 0.0],
            [5.0, 7.0, 0.0, 6.0],
            [0.0, 0.0, 6.0, 0.0],
        ]
    )

    np.testing.assert_allclose(W1, adjacency_matrix)


def test_degree_directed_graph():
    #            e1
    # graph:  n1 -> n2
    #      e2  |  /
    #          v /   (e4 (n2->n3))
    #         n3 -> n4
    #            e3

    n1 = Node(1.0)
    n2 = Node(2.0)
    n3 = Node(3.0)
    n4 = Node(4.0)

    e1 = Edge([n1, n2], weight=4.0)
    e2 = Edge([n1, n3], weight=5.0)
    e3 = Edge([n3, n4], weight=6.0)
    e4 = Edge([n2, n3], weight=7.0)

    g1 = Graph([n1, n2, n3, n4], [e1, e2, e3, e4])
    D1 = g1.get_D()

    adjacency_matrix = np.array(
        [
            [0.0, 4.0, 5.0, 0.0],
            [0.0, 0.0, 7.0, 0.0],
            [0.0, 0.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(D1, np.diag(adjacency_matrix.sum(1)))


def test_degree_undirected_graph():
    #            e1
    # graph:  n1 -- n2
    #      e2  |  /
    #          | /
    #         n3 -- n4
    #            e3

    n1 = Node(1.0)
    n2 = Node(2.0)
    n3 = Node(3.0)
    n4 = Node(4.0)

    e1 = Edge([n1, n2], weight=4.0, directed=False)
    e2 = Edge([n1, n3], weight=5.0, directed=False)
    e3 = Edge([n3, n4], weight=6.0, directed=False)
    e4 = Edge([n2, n3], weight=7.0, directed=False)

    g1 = Graph([n1, n2, n3, n4], [e1, e2, e3, e4], directed=False)
    D1 = g1.get_D()

    adjacency_matrix = np.array(
        [
            [0.0, 4.0, 5.0, 0.0],
            [4.0, 0.0, 7.0, 0.0],
            [5.0, 7.0, 0.0, 6.0],
            [0.0, 0.0, 6.0, 0.0],
        ]
    )

    np.testing.assert_allclose(D1, np.diag(adjacency_matrix.sum(1)))
    np.testing.assert_allclose(D1, np.diag(adjacency_matrix.sum(0)))
