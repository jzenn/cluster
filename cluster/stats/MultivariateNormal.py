import numpy as np
import numpy.linalg as lina
import numpy.random as rand


class MultivariateNormal:
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

        self.det_covariance = lina.det(covariance)
        self.inv_covariance = lina.inv(covariance)

        self.k = mean.shape[0]
        self.factor = 1 / np.sqrt(np.power(2 * np.pi, self.k) * self.det_covariance)

        self.L = lina.cholesky(self.covariance)

    def pdf(self, x):
        gaussian = np.exp(-1 / 2 * (x - self.mean).dot(self.inv_covariance.dot(x - self.mean)))
        return self.factor * gaussian

    def sample(self, n):
        z = rand.normal(0, 1, self.k * n).reshape(self.k, n)
        return self.L.dot(z).T + np.expand_dims(self.mean, 0).repeat(n, axis=0)
