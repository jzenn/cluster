import numpy as np
import torch

from ..globals import device


class Graph:
    def __init__(self, nodes, edges, directed=False, attributes=None, dev=None):
        if not dev:
            dev = device
        self.device = dev

        self.directed = directed
        self.attributes = attributes if attributes else {}

        self.node_list = nodes
        self.edge_list = edges

        self.nodes = {node for node in nodes}
        self.edges = {edge for edge in edges}

        self.node_edges = {node: list() for node in nodes}

        for edge in edges:
            from_node = edge.get_from_node()
            to_node = edge.get_to_node()

            self.node_edges[from_node] += [to_node]
            from_node.set_degree(len(self.node_edges[from_node]))

            if not self.directed:
                self.node_edges[to_node] += [from_node]
                to_node.set_degree(len(self.node_edges[to_node]))

        self.D = None
        self.W = None

    def add_node(self, node):
        self.node_list += [node]
        self.nodes += {node}
        self.node_edges[node] = list()

        self.D = None
        self.W = None

    def add_edge(self, edge):
        self.edge_list += edge
        self.edges += {edge}

        from_node = edge.get_from_node()
        to_node = edge.get_to_node()

        self.node_edges[from_node] += [to_node]
        from_node.set_degree(len(self.node_edges[from_node]))

        if not self.directed:
            self.node_edges[to_node] += [from_node]
            to_node.set_degree(len(self.node_edges[to_node]))

        self.D = None
        self.W = None

    def get_neighbors(self, node):
        return self.node_edges[node]

    def get_node_list(self):
        return self.node_list

    def get_edge_list(self):
        return self.edge_list

    def compute_D(self):
        if self.W is not None:
            n = len(self.node_list)
            if self.device.type == 'cpu':
                D = np.diag(np.sum(self.W, axis=0))
                self.D = D
            elif self.device.type == 'cuda':
                D = torch.diag(torch.sum(self.W, axis=0))
                D.to(self.device)
                self.D = D
        else:
            self.compute_W()
            self.compute_D()

    def compute_W(self):
        n = len(self.node_list)
        if self.device.type == 'cpu':
            W = np.zeros([n, n])
        elif self.device.type == 'cuda':
            W = torch.zeros(n, n)
            W.to(self.device)

        for edge in self.edge_list:
            from_node = edge.get_from_node().get_value()
            to_node = edge.get_to_node().get_value()

            W[from_node, to_node] = edge.get_weight()

            if not self.directed:
                W[to_node, from_node] = edge.get_weight()

        self.W = W

    def get_L(self):
        if self.D is not None and self.W is not None:
            return self.D - self.W
        else:
            self.compute_W()
            self.compute_D()
            return self.get_L()

    def __str__(self):
        nlt = '\n\t\t'
        return f'Graph(\n\tdirected: {self.directed}' \
               f'\n\tNodes: \n\t\t{nlt.join([str(node) for node in self.node_list])}' \
               f'\n\tEdges: \n\t\t{nlt.join([str(edge) for edge in self.edge_list])}\n)'
