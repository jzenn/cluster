import sys
import warnings
import numpy as np
import numpy.random as rand
import numpy.linalg as lina


class KMeansClustering:
    def __init__(self, k, max_iter=1000, rep=10, seed=None, eps=1e-10, norm=None):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.eps = eps
        self.norm = norm
        self.rep = rep

        self.points = None
        self.cluster_belongings = None
        self.objective = np.inf

    def _set_seed(self):
        if self.seed is None:
            rand.seed(rand.randint(2 ** 32 - 1))
        else:
            rand.seed(self.seed)

    def _cluster(self, points):
        self._set_seed()
        objective = np.inf

        # points are expected to be of size: number_of_points x features (N x n)
        N, n = points.shape

        # stop criterion
        stop = False
        i = 0

        # choose means randomly
        means = points[rand.randint(0, N, self.k)]
        cluster_belongings = None

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

            objective = 0
            for mean_index in range(self.k):
                objective += np.sum(
                    lina.norm(points_[cluster_belongings == mean_index, 0] - means[mean_index]) ** 2, axis=0
                )

        if i == self.max_iter:
            warnings.warn(f'reached maximum number of iterations with {self.max_iter}')
        else:
            print(f'clusters have converged in {i} iterations')

        return points, cluster_belongings, objective

    def get_points(self):
        return self.points

    def get_labels(self):
        return self.cluster_belongings

    def cluster(self, points):
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

    def get_objective(self):
        return self.objective
