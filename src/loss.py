import numpy as np


class MSE:
    def gradient(self, y, F):
        return F - y

    def hessian(self, y, F):
        return np.ones_like(y)

    def leaf_value(self, g, h):
        return -g.sum() / h.sum()

    def init_prediction(self, y):
        return np.mean(y)


class LogLoss:
    def gradient(self, y, F):
        p = 1.0 / (1.0 + np.exp(-F))
        return p - y

    def hessian(self, y, F):
        p = 1.0 / (1.0 + np.exp(-F))
        return p * (1.0 - p)

    def leaf_value(self, g, h):
        return -g.sum() / (h.sum() + 1e-9)

    def init_prediction(self, y):
        p = np.clip(np.mean(y), 1e-7, 1 - 1e-7)
        return np.log(p / (1.0 - p))
