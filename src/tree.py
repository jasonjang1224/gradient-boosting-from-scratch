import numpy as np


class ObliviousTree:
    def __init__(self, max_depth=4, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.splits = []       # list of (feature, threshold), length == max_depth
        self.leaf_values = None
        self.n_leaves = None

    def fit(self, X, g, h):
        N = len(g)
        node_indices = [np.arange(N)]  # all samples start in one root node

        for _ in range(self.max_depth):
            feature, threshold = self._find_split(X, g, h, node_indices)
            self.splits.append((feature, threshold))

            new_nodes = []
            for idx in node_indices:
                mask = X[idx, feature] <= threshold
                left_idx = idx[mask]
                right_idx = idx[~mask]
                new_nodes.append(left_idx)
                new_nodes.append(right_idx)
            node_indices = new_nodes

        self.n_leaves = 2 ** self.max_depth
        self.leaf_values = np.zeros(self.n_leaves)
        for k, idx in enumerate(node_indices):
            if len(idx) > 0:
                self.leaf_values[k] = -g[idx].sum() / (h[idx].sum() + 1e-9)

        return self

    def _find_split(self, X, g, h, node_indices):
        _, F = X.shape
        best_gain, best_f, best_t = 0.0, 0, 0.0

        for j in range(F):
            all_x = np.concatenate([X[idx, j] for idx in node_indices])
            unique_x = np.unique(all_x)
            if len(unique_x) < 2:
                continue
            thresholds = (unique_x[:-1] + unique_x[1:]) / 2.0

            for thresh in thresholds:
                gain_total = 0.0
                valid = True

                for idx in node_indices:
                    mask = X[idx, j] <= thresh
                    n_L, n_R = mask.sum(), (~mask).sum()
                    if n_L < self.min_samples_leaf or n_R < self.min_samples_leaf:
                        valid = False
                        break

                    gL = g[idx][mask].sum()
                    gR = g[idx][~mask].sum()
                    hL = h[idx][mask].sum()
                    hR = h[idx][~mask].sum()
                    gt = g[idx].sum()
                    ht = h[idx].sum()

                    gain_total += (gL ** 2 / (hL + 1e-9) + gR ** 2 / (hR + 1e-9)
                                   - gt ** 2 / (ht + 1e-9))

                if valid and gain_total > best_gain:
                    best_gain, best_f, best_t = gain_total, j, thresh

        return best_f, best_t

    def predict(self, X):
        indices = np.zeros(len(X), dtype=int)
        for feature, threshold in self.splits:
            bit = (X[:, feature] > threshold).astype(int)
            indices = indices * 2 + bit
        return self.leaf_values[indices]

    def get_leaf_indices(self, X):
        indices = np.zeros(len(X), dtype=int)
        for feature, threshold in self.splits:
            indices = indices * 2 + (X[:, feature] > threshold).astype(int)
        return indices
