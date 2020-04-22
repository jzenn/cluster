import numpy as np
import numpy.random as rand


class MixtureNormal:
    def __init__(self, multivariates, weights):
        self.multivariates = multivariates
        self.weights = weights

    def pdf(self, x):
        multivariates_eval = [m.pdf(x) for m in self.multivariates]
        return np.sum(self.weights * np.array(multivariates_eval))

    def sample(self, n):
        z = rand.rand(n)
        weights_acc = np.add.accumulate(self.weights)
        weights_acc = np.expand_dims(weights_acc, 0).reshape(weights_acc.shape[0], 1)

        number_samples_acc = np.sum(z <= weights_acc, axis=1)
        number_samples = number_samples_acc - np.concatenate([np.array([0]),
                                                              number_samples_acc[:weights_acc.shape[0] - 1]])

        samples = []
        for i in range(number_samples.shape[0]):
            m = self.multivariates[i]
            n = number_samples[i]
            samples += [m.sample(n)]

        return np.concatenate(samples)
