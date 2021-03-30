import numpy as np

from cluster.graph import Graph


# no type-hints for axis to not require the matplotlib-dependency
def visualize_graph_2d(
    points: np.array, graph: Graph, ax, title: str, annotate: bool = False
):
    """
    visualize a 2D graph in a matplotlib-figure

    :param points: points that are scattered
    :param graph: graph that is plotted (values of nodes are coordinates)
    :param ax: a matplotlib axis the data is plotted on
    :param title: a title for the figure
    :param annotate: whether to annotate the nodes with their corresponding index
    :return: matplotlib axis
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


def visualize_clusters_2d(points: np.array, labels: np.array, ax, title: str):
    """
    visualize 2D clusters in a matplotlib-figure

    :param points: the dataset
    :param labels: the labels of each point in the dataset
    :param ax: a matplotlib axis the data is plotted on
    :param title: a title for the figure
    :return: matplotlib axis
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