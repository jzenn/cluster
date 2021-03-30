import numpy as np

from ..graph import Edge
from ..graph import Node
from ..graph import Graph

from .similarity import (
    get_distance_matrix_from_similarity_matrix,
    get_similarity_matrix_from_distance_matrix,
)


def fc_graph_from_similarity_matrix(S: np.array) -> Graph:
    """
    create a fully connected graph from the similarity matrix S

    :param S: similarity matrix
    :return: graph
    :rtype: Graph
    """
    n = S.shape[0]
    nodes = []
    edges = []

    for node_index in range(n):
        node = Node(None)
        nodes += [node]

    for from_node_index in range(n):
        for to_node_index in range(from_node_index):
            weight = S[from_node_index, to_node_index]

            from_node = nodes[from_node_index]
            to_node = nodes[to_node_index]

            edge = Edge([from_node, to_node], weight=weight, directed=False)
            edges += [edge]

    return Graph(nodes=nodes, edges=edges, directed=False)


def eps_graph_from_similarity_matrix(
    S: np.array, eps: float, sigma: float = 1.0, distance: bool = False
) -> Graph:
    """
    create an epsilon-neighborhood graph from the similarity matrix S

    :param S: similarity or distance matrix (should be reflected in distance)
    :param eps: parameter for the epsilon-neighborhood graph
    :param sigma: paramter to construct the distance matrix from the similarity matrix, only used when distance=False
    :param distance: whether S is a distance matrix or a similarity matrix
    :return: epsilon graph created from the similarity matrix
    """
    if distance:
        distance_matrix = S
        similarity_matrix = get_similarity_matrix_from_distance_matrix(S, sigma=sigma)
    else:
        distance_matrix = get_distance_matrix_from_similarity_matrix(S, sigma=sigma)
        similarity_matrix = S

    n = S.shape[0]
    nodes = []

    for node_index in range(n):
        node = Node(None)
        nodes += [node]

    G = Graph(nodes=nodes, edges=[], directed=False)

    for from_node_index in range(n):
        for to_node_index in range(from_node_index + 1, n):

            similarity_weight = similarity_matrix[from_node_index, to_node_index]
            distance_weight = distance_matrix[from_node_index, to_node_index]

            if distance_weight <= eps:
                from_node = nodes[from_node_index]
                to_node = nodes[to_node_index]

                edge = Edge(
                    [from_node, to_node], weight=similarity_weight, directed=False
                )
                G.add_edge(edge)

    return G


def kNN_graph_from_similarity_matrix(S: np.array, k: int) -> Graph:
    """
    create a k-nearest-neighbor graph from similarity matrix S

    :param S: similarity matrix
    :param k: number of neighbors for each node to consider
    :return: k-nearest-neighbor graph created from the similarity matrix
    """
    n = S.shape[0]
    nodes = []

    for node_index in range(n):
        node = Node(None)
        nodes += [node]

    G = Graph(nodes=nodes, edges=[], directed=False)

    for from_node_index in range(n):
        kNN = np.argsort(-S[from_node_index, :])[: k + 1]
        kNN = np.delete(kNN, np.where(kNN == from_node_index))

        for neighbor_index in kNN:
            weight = S[from_node_index, neighbor_index]
            from_node = nodes[from_node_index]
            to_node = nodes[neighbor_index]

            edge = Edge([from_node, to_node], weight=weight, directed=False)
            G.add_edge(edge)

    return G


def mkNN_graph_from_similarity_matrix(S: np.array, k: int) -> Graph:
    """
    create a mutual k-nearest-neighbor from similarity matrix S

    :param S: the similarity matrix
    :param k: number of neighbors for each node to consider (mutually)
    :return: mutual k-nearest-neighbor graph created from the similarity matrix
    """
    n = S.shape[0]
    nodes = []

    for node_index in range(n):
        node = Node(None)
        nodes += [node]

    G = Graph(nodes=nodes, edges=[], directed=False)

    for from_node_index in range(n):
        kNN_from = np.argsort(-S[from_node_index, :])[: k + 1]
        kNN_from = np.delete(kNN_from, np.where(kNN_from == from_node_index))

        for neighbor_index in kNN_from:
            kNN_to = np.argsort(-S[:, neighbor_index])[: k + 1].T
            kNN_to = np.delete(kNN_to, np.where(kNN_to == neighbor_index))

            if from_node_index in kNN_to:
                weight = S[from_node_index, neighbor_index]
                from_node = nodes[from_node_index]
                to_node = nodes[neighbor_index]

                edge = Edge([from_node, to_node], weight=weight, directed=False)
                G.add_edge(edge)

    return G
