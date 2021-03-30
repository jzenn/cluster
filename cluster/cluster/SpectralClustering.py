import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

from typing import Union

from .KMeansClustering import KMeansClustering
from ..graph import Graph

from ..utils import (
    kNN_graph_from_similarity_matrix,
    mkNN_graph_from_similarity_matrix,
    fc_graph_from_similarity_matrix,
    eps_graph_from_similarity_matrix,
    get_similarity_matrix,
)


class SpectralClustering:
    def __init__(
        self,
        k: int,
        max_iter: int = 1000,
        rep: int = 10,
        seed: int = None,
        eps: float = 1e-10,
        norm: int = None,
        sparse: bool = True,
        graph: str = "kNN",
        graph_param: Union[int, float] = 5,
        normalized: bool = True,
        vectorize_similarity_matrix: bool = True,
    ) -> None:
        """
        Spectral Clustering object

        :param k: the number of clusters to create
        :param max_iter: maximum number of iterations to perform for convergence of clusters in k-Means iteration
        :param rep: number of times to repeat the k-Means clustering
        :param seed: seed to use
        :param eps: stopping criterion
        :param norm: the norm to use
        :param sparse: whether to use a sparse representation of the graph (using scipy.sparse)
        :param graph: one of 'kNN', 'mkNN', 'eps', 'fc'
            - 'kNN': k-Nearest-Neighbor graph
            - 'mkNN': mutual k-Nearest-Neighbor graph
            - 'eps': epsilon connected graph
            - 'fc': fully connected graph
        :param graph_param: the parameters for the specified graph (i.e.: number of neighbors, epsilon)
        :param normalized: whether to use normalized spectral clustering
        :param vectorize_similarity_matrix: whether to vectorize the matrix or filling the matrix iteratively
        :param use_scipy_eigh: whether to use scipy's eigh for computing the eigenvectors of the graph Laplacian
        """
        self.k = k
        self.max_iter = max_iter
        self.rep = rep
        self.seed = seed
        self.eps = eps
        self.norm = norm
        self.sparse = sparse
        self.graph_type = graph
        self.graph_param = graph_param
        self.normalized = normalized
        self.vectorize_similarity_matrix = vectorize_similarity_matrix

        self.kmeans = KMeansClustering(
            k=self.k,
            max_iter=self.max_iter,
            seed=self.seed,
            eps=self.eps,
            norm=self.norm,
            rep=self.rep,
        )

        self.points = None
        self.cluster_belongings = None

        self.S = None
        self.G = None
        self.L = None

        self.eigenvectors = None
        self.eigenvalues = None

    def _get_similarity_matrix(self, points: np.array) -> np.array:
        return get_similarity_matrix(
            points=points,
            norm=self.norm,
            vectorized=self.vectorize_similarity_matrix,
        )

    def _create_graph(self, S: np.array) -> np.array:
        if self.graph_type == "kNN":
            k = self.graph_param
            if k is None:
                raise RuntimeError(f"graph param k is None.")
            G = kNN_graph_from_similarity_matrix(S, k)
        elif self.graph_type == "mkNN":
            k = self.graph_param
            if k is None:
                raise RuntimeError(f"graph param k is None.")
            G = mkNN_graph_from_similarity_matrix(S, k)
        elif self.graph_type == "eps":
            eps = self.graph_param
            if eps is None:
                raise RuntimeError(f"graph param eps is None.")
            G = eps_graph_from_similarity_matrix(S, eps)
        elif self.graph_type == "fc":
            G = fc_graph_from_similarity_matrix(S)
        else:
            raise RuntimeError(f"graph type {self.graph_type} not known.")

        return G

    def _compute_first_k_eigenvectors(self) -> np.array:
        L = csr_matrix(self.L) if self.sparse else self.L
        if self.sparse:
            eigenvalues, eigenvectors = eigsh(L, k=self.k, sigma=0.0, mode="normal")
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors.T
        else:
            eigenvalues, eigenvectors = eigh(L)
            self.eigenvalues = eigenvalues[: self.k]
            self.eigenvectors = eigenvectors[:, : self.k].T

    def get_S(self) -> np.array:
        """
        return similarity matrix S

        :return: similarity matrix S
        :rtype: np.array
        """
        return self.S

    def get_graph(self) -> Graph:
        """
        return graph

        :return: graph object
        :rtype: Graph
        """
        return self.G

    def get_L(self) -> np.array:
        """
        return graph Laplacian

        :return: graph Laplacian L
        :rtype: np.array
        """
        return self.L

    def get_points(self) -> np.array:
        """
        return clustered points

        :return: points
        :rtype: np.array
        """
        return self.points

    def get_labels(self) -> np.array:
        """
        return labels

        :return:
        :rtype: np.array
        """
        return self.cluster_belongings

    def cluster(self, points: np.array) -> np.array:
        # similarity matrix
        self.S = self._get_similarity_matrix(points)

        # graph from similarity matrix
        self.G = self._create_graph(self.S)

        # graph Laplacian
        if self.normalized:
            self.L = self.G.get_L()
            D = self.G.get_D()

            with np.errstate(divide="raise"):
                try:
                    self.L = (
                        np.diag(np.diag(D) ** (-1 / 2))
                        @ self.L
                        @ np.diag(np.diag(D) ** (-1 / 2))
                    )
                except FloatingPointError:
                    raise FloatingPointError(
                        "the graph has some isolated vertices, consider switching to a kNN graph, "
                        "or consider increasing the graph parameter"
                    )

        else:
            self.L = self.G.get_L()

        # get the first k eigenvectors
        self._compute_first_k_eigenvectors()

        # cluster the eigenvectors
        if self.normalized:
            D = self.G.get_D()

            self.kmeans.cluster(np.diag(np.diag(D) ** (-1 / 2)) @ self.eigenvectors.T)
        else:
            self.kmeans.cluster(self.eigenvectors.T)

        # get the cluster belongings
        self.cluster_belongings = self.kmeans.get_labels()
