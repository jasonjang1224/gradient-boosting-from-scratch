import numpy as np

from loss import MSE
from tree import ObliviousTree
from vanilla_gbm import DecisionTree
from encoder import OrderedTargetEncoder
from boosting import VanillaBoosting, OrderedBoosting, SGLB, SGLBEnsemble


class GradientBoosting:
    def __init__(
        self,
        tree_cls=ObliviousTree,
        loss=None,
        boosting=None,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        min_samples_leaf=1,
        cat_features=None,
        prior_weight=1.0,
    ):
        self.tree_cls = tree_cls
        self.loss = loss or MSE()
        self.boosting = boosting or VanillaBoosting()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.cat_features = cat_features or []
        self.prior_weight = prior_weight

        self.F0 = None
        self.tree_sets = []
        self._ts_enc = None

    def fit(self, X, y):
        self.F0 = self.loss.init_prediction(y)

        if self.cat_features:
            self._ts_enc = OrderedTargetEncoder(
                cat_features=self.cat_features,
                prior_weight=self.prior_weight,
            ).fit(y)

        rng = np.random.default_rng(42)

        if isinstance(self.boosting, SGLBEnsemble):
            self.tree_sets = []
            for seed in self.boosting.seeds:
                sglb = self.boosting.make_sglb(seed)
                trees = self._fit_single(X, y, sglb, rng=np.random.default_rng(seed))
                self.tree_sets.append(trees)
        else:
            trees = self._fit_single(X, y, self.boosting, rng=rng)
            self.tree_sets = [trees]

        return self

    def _fit_single(self, X, y, boosting_strategy, rng):
        N = len(y)
        F_pred = np.full(N, self.F0)
        trees = []
        prev_tree = None

        for _ in range(self.n_estimators):
            sigma = rng.permutation(N)

            if self.cat_features and self._ts_enc is not None:
                X_enc = self._ts_enc.transform(X, y, sigma)
            else:
                X_enc = X

            g, h = boosting_strategy.compute_gradients(
                y, F_pred, self.loss, self.learning_rate,
                prev_tree=prev_tree, sigma=sigma, X_enc=X_enc,
            )

            tree = self.tree_cls(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            ).fit(X_enc[sigma], g[sigma], h[sigma])

            F_pred = boosting_strategy.update(
                F_pred, tree.predict(X_enc), self.learning_rate
            )
            prev_tree = tree
            trees.append(tree)

        return trees

    def _predict_single(self, X, trees):
        F = np.full(len(X), self.F0)
        for tree in trees:
            F = F + self.learning_rate * tree.predict(X)
        return F

    def predict(self, X):
        if isinstance(self.boosting, SGLBEnsemble):
            preds = np.stack(
                [self._predict_single(X, trees) for trees in self.tree_sets],
                axis=0,
            )
            return preds.mean(axis=0)
        return self._predict_single(X, self.tree_sets[0])

    def predict_uncertainty(self, X):
        if not isinstance(self.boosting, SGLBEnsemble):
            raise ValueError("predict_uncertainty requires SGLBEnsemble")
        preds = np.stack(
            [self._predict_single(X, trees) for trees in self.tree_sets],
            axis=0,
        )
        return preds.var(axis=0)
