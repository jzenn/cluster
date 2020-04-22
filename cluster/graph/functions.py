import numpy as np
import matplotlib.pyplot as plt

from .Edge import Edge
from .Node import Node
from .Graph import Graph
from ..globals import device


def fc_graph_from_similarity_matrix(S, dev=None):
    if not dev:
        dev = device

    n = S.shape[0]
    nodes = []
    edges = []

    for node_index in range(n):
        node = Node(node_index, dev=dev)
        nodes += [node]

    for from_node_index in range(n):
        for to_node_index in range(from_node_index):
            weight = S[from_node_index, to_node_index]

            from_node = nodes[from_node_index]
            to_node = nodes[to_node_index]

            edge = Edge((from_node, to_node), weight, dev=dev)
            edges += [edge]

    return Graph(nodes=nodes, edges=edges, dev=dev)


def eps_graph_from_similarity_matrix(S, eps, dev=None):
    if not dev:
        dev = device

    n = S.shape[0]
    nodes = []
    edges = []

    for node_index in range(n):
        node = Node(node_index, dev=dev)
        nodes += [node]

    for from_node_index in range(n):
        for to_node_index in range(from_node_index):

            weight = S[from_node_index, to_node_index]

            if weight <= eps:
                from_node = nodes[from_node_index]
                to_node = nodes[to_node_index]

                edge = Edge((from_node, to_node), weight, dev=dev)
                edges += [edge]

    return Graph(nodes=nodes, edges=edges, dev=dev)


def kNN_graph_from_similarity_matrix(S, k, dev=None):
    if not dev:
        dev = device

    n = S.shape[0]
    nodes = []
    edges = []

    for node_index in range(n):
        node = Node(node_index, dev=dev)
        nodes += [node]

    for from_node_index in range(n):
        kNN = np.argsort(S[from_node_index, :])[:k + 1]
        kNN = np.delete(kNN, np.where(kNN == from_node_index))

        for neighbor_index in kNN:
            weight = S[from_node_index, neighbor_index]
            from_node = nodes[from_node_index]
            to_node = nodes[neighbor_index]

            edge = Edge((from_node, to_node), weight, dev=dev)
            edges += [edge]

    return Graph(nodes=nodes, edges=edges, dev=dev)


def mkNN_graph_from_similarity_matrix(S, k, dev=None):
    if not dev:
        dev = device

    n = S.shape[0]
    nodes = []
    edges = []

    for node_index in range(n):
        node = Node(node_index, dev=dev)
        nodes += [node]

    for from_node_index in range(n):
        kNN_from = np.argsort(S[from_node_index, :])[:k + 1]
        kNN_from = np.delete(kNN_from, np.where(kNN_from == from_node_index))

        for neighbor_index in kNN_from:
            kNN_to = np.argsort(S[:, neighbor_index])[:k + 1].T
            kNN_to = np.delete(kNN_to, np.where(kNN_to == neighbor_index))

            if from_node_index in kNN_to:
                weight = S[from_node_index, neighbor_index]
                from_node = nodes[from_node_index]
                to_node = nodes[neighbor_index]

                edge = Edge((from_node, to_node), weight, dev=dev)
                edges += [edge]

    return Graph(nodes=nodes, edges=edges, dev=dev)


def visualize_graph_2d(points, graph, ax, title, annotate=True):

    for i in range(len(graph.get_edge_list())):
        edge = graph.get_edge_list()[i]
        left = points[edge.get_from_node().get_value()]
        right = points[edge.get_to_node().get_value()]

        ax.plot([left[0], right[0]], [left[1], right[1]], color='lightblue')

    ax.scatter(points[:, 0], points[:, 1], s=5, color='blue', zorder=10 * len(graph.get_edge_list()))

    if annotate:
        for node in graph.get_node_list():
            idx = node.get_value()
            ax.annotate(idx, (points[idx, 0] + 1e-2, points[idx, 1] + 1e-2))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_title(title)

    return ax
