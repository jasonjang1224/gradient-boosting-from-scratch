import numpy as np


class VanillaBoosting:
    def compute_gradients(self, y, F_pred, loss, lr,
                          prev_tree=None, sigma=None, X_enc=None):
        g = loss.gradient(y, F_pred)
        h = loss.hessian(y, F_pred)
        return g, h

    def update(self, F_pred, tree_pred, lr):
        return F_pred + lr * tree_pred


class OrderedBoosting:
    def compute_gradients(self, y, F_pred, loss, lr,
                          prev_tree=None, sigma=None, X_enc=None):
        N = len(y)
        base = loss.gradient(y, F_pred)

        if prev_tree is not None and X_enc is not None:
            leaf_ids = prev_tree.get_leaf_indices(X_enc)
            n_leaves = prev_tree.n_leaves
        else:
            leaf_ids = np.zeros(N, dtype=int)
            n_leaves = 1

        leaf_sum = np.zeros(n_leaves)
        leaf_cnt = np.zeros(n_leaves, dtype=int)
        g = np.empty(N)

        for r in range(N):
            k = sigma[r]
            li = leaf_ids[k]
            corr = leaf_sum[li] / leaf_cnt[li] if leaf_cnt[li] > 0 else 0.0
            g[k] = base[k] - corr

            leaf_sum[li] += base[k]
            leaf_cnt[li] += 1

        h = loss.hessian(y, F_pred)
        return g, h

    def update(self, F_pred, tree_pred, lr):
        return F_pred + lr * tree_pred


class SGLB:
    def __init__(self, alpha=0.01, beta=10.0, rng=None):
        self.alpha = alpha
        self.beta = beta
        self.rng = rng or np.random.default_rng()

    def compute_gradients(self, y, F_pred, loss, lr,
                          prev_tree=None, sigma=None, X_enc=None):
        g = loss.gradient(y, F_pred)
        h = loss.hessian(y, F_pred)
        sigma_noise = np.sqrt(2.0 * lr / self.beta)
        g = g + self.rng.normal(0.0, sigma_noise, size=g.shape)
        return g, h

    def update(self, F_pred, tree_pred, lr):
        return (1.0 - self.alpha * lr) * F_pred + lr * tree_pred


class SGLBEnsemble:
    def __init__(self, alpha=0.01, beta=10.0, n_chains=10):
        self.alpha = alpha
        self.beta = beta
        self.n_chains = n_chains
        self.seeds = list(range(n_chains))

    def make_sglb(self, seed):
        return SGLB(alpha=self.alpha, beta=self.beta,
                    rng=np.random.default_rng(seed))
