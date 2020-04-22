import torch

from ..globals import device


class Edge:
    def __init__(self, connection, weight=1, directed=False, dev=None):
        if not dev:
            dev = device
        self.device = dev

        self.from_node = connection[0]
        self.to_node = connection[1]

        if self.device.type == 'cpu':
            self.weight = weight
        elif self.device.type == 'cuda':
            self.weight = torch.tensor(weight)

    def get_from_node(self):
        return self.from_node

    def get_to_node(self):
        return self.to_node

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight

    def __str__(self):
        return f'Edge(({self.from_node}) -- ({self.to_node})), weight: {self.weight}'