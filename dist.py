import numpy as np

class NormalDist:
    def __init__(self, mu, sigma):
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        self.size = self.mu.size

    def draw(self):
        return np.abs(np.random.normal(self.mu, self.sigma))

    def draw_i(self, i):
        return np.abs(np.random.normal(self.mu[i], self.sigma[i]))

    def __repr__(self):
        return "(NormalDist: mu = {mu}, sigma = {sigma})".format(
            mu = self.mu,
            sigma = self.sigma
        )

class UnifDist():
    def __init__(self, mu, width):
        self.mu = np.array(mu)
        self.width = np.array(width)
        self.size = self.mu.size

    def draw(self):
        return np.abs(np.random.uniform(self.mu - self.width,
            self.mu + self.width))

    def draw_i(self, i):
        return np.abs(np.random.normal(self.mu[i] - self.width[i],
            self.mu[i] + self.width[i]))

    def __repr__(self):
        return "(UnifDist: mu = {mu}, width = {width})".format(
            mu = self.mu,
            width = self.width
        )

class GammaDist():
    def __init__(self, k, theta):
        self.k = np.array(k)
        self.theta = np.array(theta)
        self.size = self.k.size
        self.shape = self.k.shape

    def draw(self):
        return np.random.gamma(self.k, self.theta)

    def draw_row(self, i):
        return np.random.gamma(self.k[i], self.theta[i])

    def draw_elem(self, x):
        assert x >= 0
        i = x // self.shape[1]
        j = x % self.shape[1]
        return np.random.gamma(self.k[i][j], self.theta[i][j])

    def draw_i_j(self, i, j):
        return np.random.gamma(self.k[i][j], self.theta[i][j])

    def __repr__(self):
        return "(GammaDist: k = {k}, width = {theta})".format(
            k = self.k,
            theta = self.theta
        )
