from typing import List
from .Node import Node


class Edge:
    def __init__(
        self, connection: List, directed: bool = True, weight: float = 1.0
    ) -> None:
        """
        Edge between two nodes in a graph

        :param connection: list of two nodes where the ordering indicates the direction of the edge (if directed=True)
        :param directed: indicating whether this edge is directed (from_node -> to_node)
        :param weight: the weight of the edge
        """
        self.from_node: Node = connection[0]
        self.to_node: Node = connection[1]

        self.directed = directed
        self.weight = weight

        if self.directed:
            self.from_node.set_attribute(
                "out_degree", self.from_node.get_attribute("out_degree") + 1
            )
            self.to_node.set_attribute(
                "in_degree", self.to_node.get_attribute("in_degree") + 1
            )

            self.from_node.increase_degree(by=1)
            self.to_node.increase_degree(by=1)
        else:
            self.from_node.set_attribute("out_degree", None)
            self.from_node.set_attribute("in_degree", None)
            self.to_node.set_attribute("out_degree", None)
            self.to_node.set_attribute("in_degree", None)

            self.from_node.increase_degree(by=1)
            self.to_node.increase_degree(by=1)

    def get_from_node(self) -> Node:
        """
        get the from-node (where the edge starts from)

        :return: from-node
        :rtype: Node
        """
        return self.from_node

    def get_to_node(self) -> Node:
        """
        get the to-node (where the edge is directed to)

        :return: to-node
        :rtype: Node
        """
        return self.to_node

    def get_weight(self) -> float:
        """
        get the weight of the edge

        :return: weight of the edge
        :rtype: float
        """
        return self.weight

    def set_weight(self, weight: float) -> None:
        """
        set the weight of the edge

        :param weight: weight of the edge
        :return: None
        """
        self.weight = weight

    def __str__(self) -> str:
        return f"Edge(({self.from_node}) -- ({self.to_node})), weight: {self.weight}"
