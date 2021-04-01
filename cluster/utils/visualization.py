import numpy as np

from cluster.graph import Graph


# no type-hints for axis to not require the matplotlib-dependency
def visualize_graph_2d(
    points: np.array, graph: Graph, ax, title: str = '', annotate: bool = False
):
    """
    visualize a 2D graph in a matplotlib-axis

    :param points: points that are scattered
    :param graph: graph that is plotted
    :param ax: a matplotlib axis the data is plotted on
    :param title: a title for the figure
    :param annotate: whether to annotate the nodes with their corresponding index
    :return: matplotlib axis

    Example
    ::

        from cluster.utils import visualize_graph_2d
        from cluster.cluster import SpectralClustering

        # cluster the dataset
        spectral_clustering = SpectralClustering(k=2, graph_param=5)
        spectral_clustering.cluster(points=points)
        labels = spectral_clustering.get_labels()

        # visualize the graph
        fig, ax = plt.subplots()
        visualize_graph_2d(points, spectral_clustering.get_graph(), ax, 'kNN graph with 5 neighbors')
        plt.show()

    .. plot::

        import numpy as np
        from cluster.utils import visualize_graph_2d
        from cluster.cluster import SpectralClustering

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

        # cluster the dataset
        spectral_clustering = SpectralClustering(k=2, graph_param=5)
        spectral_clustering.cluster(points=points)
        labels = spectral_clustering.get_labels()

        # visualize the graph
        fig, ax = plt.subplots()
        visualize_graph_2d(points, spectral_clustering.get_graph(), ax, 'kNN graph with 5 neighbors')
        plt.show()
    """
    for i in range(len(graph.get_edge_list())):
        edge = graph.get_edge_list()[i]
        left = points[edge.get_from_node().get_attribute("node_index")]
        right = points[edge.get_to_node().get_attribute("node_index")]

        ax.plot([left[0], right[0]], [left[1], right[1]], color="lightblue")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=5,
        color="blue",
        zorder=10 * len(graph.get_edge_list()),
    )

    if annotate:
        for node in graph.get_node_list():
            idx = node.get_attribute("node_index")
            ax.annotate(idx, (points[idx, 0] + 1e-2, points[idx, 1] + 1e-2))

    ax.set_title(title)

    return ax


def visualize_clusters_2d(points: np.array, labels: np.array, ax, title: str = ''):
    """
    visualize 2D clusters in a matplotlib-figure

    :param points: points that are scattered (the dataset)
    :param labels: labels of the points in the dataset
    :param ax: a matplotlib axis the data is plotted on
    :param title: a title for the figure
    :return: matplotlib axis

    Example
    ::

        from cluster.utils import visualize_clusters_2d
        from cluster.cluster import SpectralClustering

        # cluster the dataset
        spectral_clustering = SpectralClustering(k=2, graph_param=5)
        spectral_clustering.cluster(points=points)
        labels = spectral_clustering.get_labels()

        # visualize the dataset in 2D
        fig, ax = plt.subplots()
        ax = visualize_clusters_2d(points, labels, ax, 'SpectralClustering with k = 2')
        plt.show()

    .. plot::

        import numpy as np
        from cluster.utils import visualize_clusters_2d
        from cluster.cluster import SpectralClustering

        np.random.seed(42)

        N1 = 25
        N2 = 75
        r1 = 1 / 4
        r2 = 3 / 4

        x1 = 2 * np.pi * np.random.rand(N1)
        x2 = 2 * np.pi * np.random.rand(N2)
        r1 = r1 + np.random.rand(N1) * 0.1
        r2 = r2 + np.random.rand(N2) * 0.1

        points = np.vstack([np.concatenate([r1 * np.cos(x1), r2 * np.cos(x2)]),
                            np.concatenate([r1 * np.sin(x1), r2 * np.sin(x2)])]).T

        # cluster the dataset
        spectral_clustering = SpectralClustering(k=2, graph_param=5)
        spectral_clustering.cluster(points=points)
        labels = spectral_clustering.get_labels()

        # visualize the dataset in 2D
        fig, ax = plt.subplots()
        ax = visualize_clusters_2d(points, labels, ax, 'SpectralClustering with k = 2')
        plt.show()
    """
    cluster_indices = np.unique(labels)

    colors = [
        "red",
        "green",
        "blue",
        "lightgreen",
        "lightblue",
        "pink",
        "yellow",
        "orange",
    ]

    for cluster_index in cluster_indices:
        ax.scatter(
            points[labels == cluster_index, 0],
            points[labels == cluster_index, 1],
            color=colors[cluster_index],
            s=10,
        )

    ax.set_title(title)

    return ax
