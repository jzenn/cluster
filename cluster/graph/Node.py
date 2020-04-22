import torch

from ..globals import device


class Node:
    def __init__(self, value, attributes=None, dev=None):
        if not dev:
            dev = device
        self.device = dev

        self.degree = 0

        if self.device.type == 'cpu':
            self.value = value
        elif self.device.type == 'cuda':
            self.value = torch.tensor(value)

        self.attributes = attributes if attributes else {'degree': 0}

    def get_value(self):
        return self.value

    def get_attributes(self):
        return self.attributes

    def get_attribute(self, attribute):
        return self.attributes[attribute]

    def get_degree(self):
        return self.degree

    def set_degree(self, degree):
        self.degree = degree
        self.attributes['degree'] = degree

    def __str__(self):
        return f'Node({self.value}){", " if self.attributes else ""}' \
               f'{", ".join([str((key, value)) for key, value in self.attributes.items()])}'