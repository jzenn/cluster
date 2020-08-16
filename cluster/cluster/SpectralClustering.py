import torch
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

from .KMeansClustering import KMeansClustering
from ..linalg import get_similarity_matrix
from ..graph import kNN_graph_from_similarity_matrix, mkNN_graph_from_similarity_matrix, \
    fc_graph_from_similarity_matrix, eps_graph_from_similarity_matrix
from ..linalg import get_first_k_eig


class SpectralClustering:
    def __init__(self, k, max_iter=1000, seed=None, eps=1e-10, norm=None, device='cpu', sparse=True, graph='kNN',
                 graph_param=5, normalized=True, vectorize_similarity_matrix=True, use_scipy_eigh=False):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.eps = eps
        self.norm = norm
        self.device = torch.device(device)
        self.sparse = sparse
        self.graph_type = graph
        self.graph_param = graph_param
        self.normalized = normalized
        self.vectorize_similarity_matrix = vectorize_similarity_matrix
        self.use_scipy_eigh = use_scipy_eigh

        self.kmeans = KMeansClustering(k=self.k, max_iter=self.max_iter, seed=self.seed, eps=self.eps, norm=self.norm)

        self.points = None
        self.cluster_belongings = None

        self.S = None
        self.G = None
        self.L = None

        self.eigenvectors = None
        self.eigenvalues = None

    def _get_similarity_matrix(self, points):
        return get_similarity_matrix(points=points, dev=self.device, norm=self.norm,
                                     vectorized=self.vectorize_similarity_matrix)

    def _create_graph(self, S):
        if self.graph_type == 'kNN':
            k = self.graph_param
            if k is None:
                raise RuntimeError(f'graph param k is None.')
            G = kNN_graph_from_similarity_matrix(S, k, dev=self.device)
        elif self.graph_type == 'mkNN':
            k = self.graph_param
            if k is None:
                raise RuntimeError(f'graph param k is None.')
            G = mkNN_graph_from_similarity_matrix(S, k, dev=self.device)
        elif self.graph_type == 'eps':
            eps = self.graph_param
            if eps is None:
                raise RuntimeError(f'graph param eps is None.')
            G = eps_graph_from_similarity_matrix(S, eps, dev=self.device)
        elif self.graph_type == 'fc':
            G = fc_graph_from_similarity_matrix(S, dev=self.device)
        else:
            raise RuntimeError(f'graph type {self.graph_type} not known.')

        return G

    def _compute_first_k_eigenvectors(self):
        L = csr_matrix(self.L) if self.sparse else self.L
        if self.use_scipy_eigh:
            if self.sparse:
                eigenvalues, eigenvectors = eigsh(L, k=self.k, sigma=0., mode='normal')
                self.eigenvalues = eigenvalues
                self.eigenvectors = eigenvectors.T
            else:
                eigenvalues, eigenvectors = eigh(L)
                self.eigenvalues = eigenvalues[:self.k]
                self.eigenvectors = eigenvectors[:, :self.k].T
        else:
            self.eigenvectors, self.eigenvalues = get_first_k_eig(L, self.k, sparse=self.sparse, norm=self.norm)

    def get_S(self):
        return self.S

    def get_graph(self):
        return self.G

    def get_L(self):
        return self.L

    def get_points(self):
        return self.points

    def get_labels(self):
        return self.cluster_belongings

    def cluster(self, points):
        # similarity matrix
        self.S = self._get_similarity_matrix(points)

        # graph from similarity matrix
        self.G = self._create_graph(self.S)

        # graph Laplacian
        if self.normalized:
            self.L = self.G.get_L()
            D = self.G.get_D()

            with np.errstate(divide='raise'):
                try:
                    self.L = np.diag(np.diag(D) ** (-1/2)) @ self.L @ np.diag(np.diag(D) ** (-1/2))
                except FloatingPointError:
                    print('the graph has some isolated vertices, stopping')
                    print('consider using a kNN or mkNN graph to overcome this issue')
                    raise FloatingPointError('the graph has some isolated vertices, switch to a kNN/mkNN graph')

        else:
            self.L = self.G.get_L()

        # get the first k eigenvectors
        self._compute_first_k_eigenvectors()

        # cluster the eigenvectors
        if self.normalized:
            D = self.G.get_D()

            self.kmeans.cluster(np.diag(np.diag(D) ** (-1/2)) @ self.eigenvectors.T)
        else:
            self.kmeans.cluster(self.eigenvectors.T)

        # get the cluster belongings
        self.cluster_belongings = self.kmeans.get_labels()
