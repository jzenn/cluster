import numpy as np

from ..graph import visualize_graph_2d


def visualize_clusters_2d(points, clusters, ax, title):
    cluster_indices = np.unique(clusters)

    colors = ['red', 'green', 'blue', 'lightgreen', 'lightblue', 'pink', 'yellow', 'orange']

    for cluster_index in cluster_indices:
        ax.scatter(points[clusters == cluster_index, 0],
                   points[clusters == cluster_index, 1],
                   color=colors[cluster_index], s=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_title(title)

    return ax


def visualize_graph_clusters_2d(points, spectral_clustering_instance, ax, titles, annotate=False):
    labels = spectral_clustering_instance.get_labels()
    graph = spectral_clustering_instance.get_graph()

    ax[0] = visualize_graph_2d(points, graph, ax[0], titles[0], annotate=annotate)
    ax[1] = visualize_clusters_2d(points, labels, ax[1], titles[1])

    return ax