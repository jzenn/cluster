import torch
import numpy as np

from ..globals import device


def get_similarity_matrix(points, dev=None, norm=None):
    if not dev:
        dev = device

    # expect points to be (N x k)
    if dev.type == 'cuda':
        # points.to(dev)
        M = points.unsqueeze(0).repeat(points.shape[0], 1, 1)
        S = torch.norm((M - torch.transpose(M, 0, 1)), dim=(2, 2, 2), p=norm)

    elif dev.type == 'cpu':
        M = np.expand_dims(points, 0).repeat(points.shape[0], 0)
        S = np.linalg.norm((M - np.transpose(M, (1, 0, 2))), axis=2, ord=norm)

    else:
        raise RuntimeError(f'device {device} not known.')

    return S
