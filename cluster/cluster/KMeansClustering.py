import numpy as np
import numpy.random as rand
import numpy.linalg as lina

from warnings import warn


class KMeansClustering:
    def __init__(
        self,
        k: int,
        max_iter: int = 1000,
        rep: int = 10,
        seed: int = None,
        eps: float = 1e-10,
        norm: int = 2,
    ) -> None:
        """
        k-Means Clustering object

        :param k: the number of clusters to create
        :param max_iter: maximum number of iterations to perform for convergence of clusters
        :param rep: number of times to repeat the clustering algorithm
        :param seed: seed to use
        :param eps: stopping criterion
        :param norm: the norm to use
        """
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.eps = eps
        self.norm = norm
        self.rep = rep

        self.points = None
        self.cluster_belongings = None
        self.objective = np.inf

    def _set_seed(self) -> None:
        if self.seed is None:
            rand.seed(rand.randint(2 ** 32 - 1))
        else:
            rand.seed(self.seed)

    def _get_objective(self) -> float:
        return self.objective

    def _cluster(self, points: np.array) -> (np.array, np.array, float):
        self._set_seed()

        # points are expected to be of size: number_of_points x features (N x n)
        N, n = points.shape

        # stop criterion
        stop = False
        i = 0

        # choose means randomly
        means = points[rand.randint(0, N, self.k)]
        cluster_belongings = None
        points_ = None

        while not stop and i < self.max_iter:
            # assign points to clusters
            means_ = np.expand_dims(means, 0).repeat(N, 0)
            points_ = np.expand_dims(points, 1).repeat(self.k, 1)

            dist = lina.norm(means_ - points_, axis=2, ord=self.norm)
            cluster_belongings = np.argmin(dist, axis=1)

            means_replace = np.zeros(means.shape) - 1
            for mean_index in range(self.k):
                m = np.mean(points_[cluster_belongings == mean_index, 0], axis=0)
                means_replace[mean_index, :] = m

            # stop criterion
            i += 1
            if lina.norm(means_replace - means, ord=self.norm) < self.eps:
                stop = True

            means = means_replace

        # compute objective
        objective = 0
        for mean_index in range(self.k):
            mean_index_cluster_belongings = (
                points_[cluster_belongings == mean_index, 0] - means[mean_index]
            )

            # workaround for numpy bug of lina.norm([](array of shape 0x2), ord=2)
            if np.prod(mean_index_cluster_belongings.shape) == 0:
                mean_index_cluster_belongings = np.array([])

            objective += np.sum(
                lina.norm(
                    mean_index_cluster_belongings,
                    ord=self.norm,
                )
                ** 2,
                axis=0,
            )

        if i == self.max_iter:
            warn(f"reached maximum number of iterations with {self.max_iter}")
        else:
            print(f"clusters have converged in {i} iterations")

        return points, cluster_belongings, objective

    def get_points(self) -> np.array:
        """
        get points

        :return: points
        :rtype: np.array
        """
        return self.points

    def get_labels(self) -> np.array:
        """
        get labels

        :return: labels
        :rtype: np.array
        """
        return self.cluster_belongings

    def cluster(self, points: np.array) -> None:
        """
        cluster the points provided

        :param points: the dataset to be clustered in form (N x d) where N is the number of points to be clustered
        :return: None
        """
        if self.seed is None:
            for _ in range(self.rep):
                points_, cluster_belongings_, objective_ = self._cluster(points)
                if objective_ < self.objective:
                    self.cluster_belongings = cluster_belongings_
                    self.points = points_
                    self.objective = objective_
        else:
            points_, cluster_belongings_, objective_ = self._cluster(points)
            self.points = points_
            self.cluster_belongings = cluster_belongings_
            self.objective = objective_
