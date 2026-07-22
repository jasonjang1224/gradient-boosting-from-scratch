import numpy as np


class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=4, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, g, h=None):
        self._h = h
        self.root = self._build(X, g, h if h is not None else np.ones_like(g), depth=0)
        return self

    def _build(self, X, g, h, depth):
        if depth >= self.max_depth or len(g) < 2 * self.min_samples_leaf:
            return self._make_leaf(g, h)

        feature, threshold, gain = self._find_split(X, g)
        if feature is None or gain <= 0:
            return self._make_leaf(g, h)

        mask = X[:, feature] <= threshold
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return self._make_leaf(g, h)

        node = Node()
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[mask], g[mask], h[mask], depth + 1)
        node.right = self._build(X[~mask], g[~mask], h[~mask], depth + 1)
        return node

    def _make_leaf(self, g, h):
        node = Node()
        node.value = -g.sum() / (h.sum() + 1e-9)
        return node

    def _find_split(self, X, g):
        N, F = X.shape
        sum_t = g.sum()
        score_no_split = sum_t ** 2 / N
        best_gain, best_feature, best_threshold = 0.0, None, None

        for j in range(F):
            order = np.argsort(X[:, j])
            g_sorted = g[order]
            x_sorted = X[order, j]
            sum_L = 0.0

            for i in range(1, N):
                sum_L += g_sorted[i - 1]
                sum_R = sum_t - sum_L
                n_L, n_R = i, N - i

                if x_sorted[i] == x_sorted[i - 1]:
                    continue
                if n_L < self.min_samples_leaf or n_R < self.min_samples_leaf:
                    continue

                gain = sum_L ** 2 / n_L + sum_R ** 2 / n_R - score_no_split
                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = (x_sorted[i - 1] + x_sorted[i]) / 2.0

        return best_feature, best_threshold, best_gain

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)
