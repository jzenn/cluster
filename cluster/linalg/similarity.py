import torch
import numpy as np

from ..globals import device


def get_similarity_matrix(points, sigma=1, dev=None, norm=None, vectorized=True):
    if not dev:
        dev = device

    # expect points to be (N x k)
    N, k = points.shape
    if vectorized:
        if dev.type == 'cuda':
            # points.to(dev)
            M = points.unsqueeze(0).repeat(N, 1, 1)
            S = torch.norm((M - torch.transpose(M, 0, 1)), dim=(2, 2, 2), p=norm)

        elif dev.type == 'cpu':
            M = np.expand_dims(points, 0).repeat(N, 0)
            S = np.linalg.norm((M - np.transpose(M, (1, 0, 2))), axis=2, ord=norm)

        else:
            raise RuntimeError(f'device {device} not known.')
    else:
        S = np.zeros([N, N])
        for i in range(N):
            for j in range(N):
                p = points[i, :]
                q = points[j, :]

                if dev.type == 'cuda':
                    S[i, j] = torch.norm(p - q, p=norm)
                elif dev.type == 'cpu':
                    S[i, j] = np.linalg.norm(p - q, ord=norm)
                else:
                    raise RuntimeError(f'device {device} not known.')

    # apply Gaussian kernel to distance matrix to get similarity
    return np.exp(- S ** 2) / (2 ** sigma ** 2)
