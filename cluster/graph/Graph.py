import numpy as np

from typing import List, Dict, Any, AnyStr

from .Node import Node
from .Edge import Edge


class Graph:
    def __init__(
        self,
        nodes: List[Node],
        edges: List[Edge],
        directed: bool = True,
        attributes: Dict[AnyStr, Any] = None,
    ) -> None:
        """
        Graph object consisting of nodes and edges

        :param nodes: list of nodes
        :param edges: list of edges
        :param directed: whether the given graph is directed
        :param attributes: attributes of any kind added to this graph
        """
        self.directed = directed
        self.attributes = attributes if attributes is not None else {}

        self.node_list = []
        self.edge_list = []

        self.node_edges = {node: list() for node in nodes}
        self.next_node_index = 0

        for edge in edges:
            self.add_edge(edge)

        for node in nodes:
            self.add_node(node)

        self.D = None
        self.W = None

    def add_node(self, node: Node) -> None:
        """
        add a node to the graph

        :param node: the node to be added
        :return: None
        """
        if node in self.node_list:
            return

        node.set_attribute("node_index", self.next_node_index)
        self.next_node_index += 1

        self.node_list += [node]
        self.node_edges[node] = list()

        self.D = None
        self.W = None

    def add_edge(self, edge: Edge) -> None:
        """
        add an edge to the graph

        :param edge: the edge to be added
        :return: None
        """
        if edge.directed and not self.directed:
            raise RuntimeError("Can't add a directed edge to an undirected graph")
        if not edge.directed and self.directed:
            raise RuntimeError("Can't add an un-directed edge to a directed graph")

        self.edge_list += [edge]

        # the nodes of the edge
        from_node = edge.get_from_node()
        to_node = edge.get_to_node()

        # add nodes to graph
        self.add_node(from_node)
        self.add_node(to_node)

        self.node_edges[from_node] += [to_node]

        if not self.directed:
            # expand graph's list
            self.node_edges[to_node] += [from_node]

        # if an edge is added we need to re-compute D and W (lazily)
        self.D = None
        self.W = None

    def get_neighbors(self, node: Node) -> List[Node]:
        """
        get all neighboring nodes of a node

        :param node: node to get the neighbors of
        :return: list of neighboring nodes
        :rtype: list(Node)
        """
        return self.node_edges[node]

    def get_node_list(self) -> List[Node]:
        """
        get all nodes

        :return: list of nodes of the graph
        :rtype: list(Node)
        """
        return self.node_list

    def get_edge_list(self) -> List[Edge]:
        """
        get all edges of the graph

        :return: list of edges
        :rtype: list(Edge)
        """
        return self.edge_list

    def _compute_D(self) -> None:
        """
        compute the degree matrix D

        :return: None
        """
        if self.W is not None:
            D = np.diag(np.sum(self.W, axis=1))
            self.D = D
        else:
            self._compute_W()
            self._compute_D()

    def _compute_W(self) -> None:
        """
        compute weighted adjacency matrix W

        :return: None
        """
        n = len(self.node_list)
        W = np.zeros([n, n])

        for edge in self.edge_list:
            from_node = edge.get_from_node().get_attribute("node_index")
            to_node = edge.get_to_node().get_attribute("node_index")

            W[from_node, to_node] = edge.get_weight()

            if not self.directed:
                W[to_node, from_node] = edge.get_weight()

        self.W = W

    def get_L(self) -> np.array:
        """
        get the graph Laplacian L

        :return: graph Laplacian L
        :rtype: np.array
        """
        if self.D is not None and self.W is not None:
            return self.D - self.W
        else:
            self._compute_W()
            self._compute_D()
            return self.get_L()

    def get_W(self) -> np.array:
        """
        get the weighted adjacency matrix W

        :return: weighted adjacency matrix W
        :rtype: np.array
        """
        if self.W is None:
            _ = self.get_L()
        return self.W

    def get_D(self) -> np.array:
        """
        get the degree matrix D

        :return: degree matrix D
        :rtype: np.array
        """
        if self.D is None:
            _ = self.get_L()
        return self.D

    def __str__(self) -> str:
        nlt = "\n\t\t"
        return (
            f"Graph(\n\tdirected: {self.directed}"
            f"\n\tNodes: \n\t\t{nlt.join([str(node) for node in self.node_list])}"
            f"\n\tEdges: \n\t\t{nlt.join([str(edge) for edge in self.edge_list])}\n)"
        )
