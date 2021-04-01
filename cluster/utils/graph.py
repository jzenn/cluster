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
    :return: fully connected graph
    :rtype: Graph

    Example
    ::
        import numpy as np
        from cluster.utils import fc_graph_from_similarity_matrix, visualize_graph_2d, get_similarity_matrix

        np.random.seed(42)

        N1 = 10
        N2 = 10
        r1 = 1 / 4
        r2 = 3 / 4

        x1 = 2 * np.pi * np.random.rand(N1)
        x2 = 2 * np.pi * np.random.rand(N2)
        r1 = r1 + np.random.rand(N1) * 0.1
        r2 = r2 + np.random.rand(N2) * 0.1

        points = np.vstack([np.concatenate([r1 * np.cos(x1), r2 * np.cos(x2)]),
                            np.concatenate([r1 * np.sin(x1), r2 * np.sin(x2)])]).T

        fig, ax = plt.subplots()
        visualize_graph_2d(points, fc_graph_from_similarity_matrix(get_similarity_matrix(points)), ax, 'fc graph')
        plt.show()

    .. plot::

        import numpy as np
        from cluster.utils import fc_graph_from_similarity_matrix, visualize_graph_2d, get_similarity_matrix

        np.random.seed(42)

        N1 = 10
        N2 = 10
        r1 = 1 / 4
        r2 = 3 / 4

        x1 = 2 * np.pi * np.random.rand(N1)
        x2 = 2 * np.pi * np.random.rand(N2)
        r1 = r1 + np.random.rand(N1) * 0.1
        r2 = r2 + np.random.rand(N2) * 0.1

        points = np.vstack([np.concatenate([r1 * np.cos(x1), r2 * np.cos(x2)]),
                            np.concatenate([r1 * np.sin(x1), r2 * np.sin(x2)])]).T

        fig, ax = plt.subplots()
        visualize_graph_2d(points, fc_graph_from_similarity_matrix(get_similarity_matrix(points)), ax, 'fc graph')
        plt.show()

    |


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

    :param S: similarity or distance matrix (should be reflected in parameter distance)
    :param eps: parameter for the epsilon-neighborhood graph
    :param sigma: parameter to construct the distance matrix from the similarity matrix, only used when distance=False
    :param distance: whether S is a distance matrix or a similarity matrix
    :return: epsilon-neighborhood graph
    :rtype: Graph

    Example
    ::

        import numpy as np
        from cluster.utils import eps_graph_from_similarity_matrix, visualize_graph_2d, get_similarity_matrix

        np.random.seed(42)

        N1 = 10
        N2 = 10
        r1 = 1 / 4
        r2 = 3 / 4

        x1 = 2 * np.pi * np.random.rand(N1)
        x2 = 2 * np.pi * np.random.rand(N2)
        r1 = r1 + np.random.rand(N1) * 0.1
        r2 = r2 + np.random.rand(N2) * 0.1

        points = np.vstack([np.concatenate([r1 * np.cos(x1), r2 * np.cos(x2)]),
                            np.concatenate([r1 * np.sin(x1), r2 * np.sin(x2)])]).T

        fig, ax = plt.subplots()
        visualize_graph_2d(points, eps_graph_from_similarity_matrix(get_similarity_matrix(points), eps=0.2), ax,
        'eps 0.2 graph')
        plt.show()

    .. plot::

        import numpy as np
        from cluster.utils import eps_graph_from_similarity_matrix, visualize_graph_2d, get_similarity_matrix

        np.random.seed(42)

        N1 = 10
        N2 = 10
        r1 = 1 / 4
        r2 = 3 / 4

        x1 = 2 * np.pi * np.random.rand(N1)
        x2 = 2 * np.pi * np.random.rand(N2)
        r1 = r1 + np.random.rand(N1) * 0.1
        r2 = r2 + np.random.rand(N2) * 0.1

        points = np.vstack([np.concatenate([r1 * np.cos(x1), r2 * np.cos(x2)]),
                            np.concatenate([r1 * np.sin(x1), r2 * np.sin(x2)])]).T

        fig, ax = plt.subplots()
        visualize_graph_2d(points, eps_graph_from_similarity_matrix(get_similarity_matrix(points), eps=0.5), ax,
        'eps 0.5 graph')
        plt.show()

    |



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
    :return: k-nearest-neighbor graph
    :rtype: Graph

    Example
    ::

        import numpy as np
        from cluster.utils import kNN_graph_from_similarity_matrix, visualize_graph_2d, get_similarity_matrix

        np.random.seed(42)

        N1 = 10
        N2 = 10
        r1 = 1 / 4
        r2 = 3 / 4

        x1 = 2 * np.pi * np.random.rand(N1)
        x2 = 2 * np.pi * np.random.rand(N2)
        r1 = r1 + np.random.rand(N1) * 0.1
        r2 = r2 + np.random.rand(N2) * 0.1

        points = np.vstack([np.concatenate([r1 * np.cos(x1), r2 * np.cos(x2)]),
                            np.concatenate([r1 * np.sin(x1), r2 * np.sin(x2)])]).T

        fig, ax = plt.subplots()
        visualize_graph_2d(points, kNN_graph_from_similarity_matrix(get_similarity_matrix(points), k=3), ax,
        '3-NN graph')
        plt.show()

    .. plot::

        import numpy as np
        from cluster.utils import kNN_graph_from_similarity_matrix, visualize_graph_2d, get_similarity_matrix

        np.random.seed(42)

        N1 = 10
        N2 = 10
        r1 = 1 / 4
        r2 = 3 / 4

        x1 = 2 * np.pi * np.random.rand(N1)
        x2 = 2 * np.pi * np.random.rand(N2)
        r1 = r1 + np.random.rand(N1) * 0.1
        r2 = r2 + np.random.rand(N2) * 0.1

        points = np.vstack([np.concatenate([r1 * np.cos(x1), r2 * np.cos(x2)]),
                            np.concatenate([r1 * np.sin(x1), r2 * np.sin(x2)])]).T

        fig, ax = plt.subplots()
        visualize_graph_2d(points, kNN_graph_from_similarity_matrix(get_similarity_matrix(points), k=3), ax,
        '3-NN graph')
        plt.show()

    |



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
    :return: mutual k-nearest-neighbor graph
    :rtype: Graph

    Example
    ::

        import numpy as np
        from cluster.utils import mkNN_graph_from_similarity_matrix, visualize_graph_2d, get_similarity_matrix

        np.random.seed(42)

        N1 = 10
        N2 = 10
        r1 = 1 / 4
        r2 = 3 / 4

        x1 = 2 * np.pi * np.random.rand(N1)
        x2 = 2 * np.pi * np.random.rand(N2)
        r1 = r1 + np.random.rand(N1) * 0.1
        r2 = r2 + np.random.rand(N2) * 0.1

        points = np.vstack([np.concatenate([r1 * np.cos(x1), r2 * np.cos(x2)]),
                            np.concatenate([r1 * np.sin(x1), r2 * np.sin(x2)])]).T

        fig, ax = plt.subplots()
        visualize_graph_2d(points, mkNN_graph_from_similarity_matrix(get_similarity_matrix(points), k=7), ax,
        'mutual 7-NN graph')
        plt.show()

    .. plot::

        import numpy as np
        from cluster.utils import mkNN_graph_from_similarity_matrix, visualize_graph_2d, get_similarity_matrix

        np.random.seed(42)

        N1 = 10
        N2 = 10
        r1 = 1 / 4
        r2 = 3 / 4

        x1 = 2 * np.pi * np.random.rand(N1)
        x2 = 2 * np.pi * np.random.rand(N2)
        r1 = r1 + np.random.rand(N1) * 0.1
        r2 = r2 + np.random.rand(N2) * 0.1

        points = np.vstack([np.concatenate([r1 * np.cos(x1), r2 * np.cos(x2)]),
                            np.concatenate([r1 * np.sin(x1), r2 * np.sin(x2)])]).T

        fig, ax = plt.subplots()
        visualize_graph_2d(points, mkNN_graph_from_similarity_matrix(get_similarity_matrix(points), k=7), ax,
        'mutual 7-NN graph')
        plt.show()

    |



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
