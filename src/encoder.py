import numpy as np
from collections import defaultdict


class OrderedTargetEncoder:
    def __init__(self, cat_features=None, prior_weight=1.0):
        self.cat_features = cat_features or []
        self.prior_weight = prior_weight
        self.prior_ = None

    def fit(self, y):
        self.prior_ = np.mean(y)
        return self

    def transform(self, X, y, sigma):
        N = len(y)
        X_enc = X.astype(float).copy()
        p, a = self.prior_, self.prior_weight

        for col in self.cat_features:
            cat_sum = defaultdict(float)
            cat_cnt = defaultdict(int)

            for r in range(N):
                k = sigma[r]
                cat_val = int(X[k, col])

                cnt = cat_cnt[cat_val]
                s = cat_sum[cat_val]
                X_enc[k, col] = (s + p * a) / (cnt + a)

                cat_sum[cat_val] += y[k]
                cat_cnt[cat_val] += 1

        return X_enc
