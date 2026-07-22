"""
Gradient Boosting from Scratch — Experiment
============================================
Reproduces the 6-configuration comparison from catboost_v3.tex:

  y = x0^2 + 0.5*x1 + eps,  x ~ Uniform(-3,3)^2,  eps ~ N(0, 0.09)
  N=400 (train 300, test 100), T=100, lr=0.1, depth=4

Configs:
  1. DecisionTree  + MSE      + Vanilla
  2. ObliviousTree + MSE      + Vanilla
  3. ObliviousTree + MSE      + Ordered
  4. ObliviousTree + LogLoss  + Ordered   (target binarized at median)
  5. ObliviousTree + MSE      + SGLB      (beta=10)
  6. ObliviousTree + MSE      + SGLBEnsemble (10 chains, beta=10)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from loss import MSE, LogLoss
from tree import ObliviousTree
from vanilla_gbm import DecisionTree
from boosting import VanillaBoosting, OrderedBoosting, SGLB, SGLBEnsemble
from model import GradientBoosting


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    N = 400
    X = rng.uniform(-3, 3, size=(N, 2))
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] + rng.normal(0, 0.3, size=N)
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]
    return X_train, y_train, X_test, y_test


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def run_experiment():
    X_train, y_train, X_test, y_test = make_data()
    median = np.median(y_train)
    y_train_bin = (y_train > median).astype(float)
    y_test_bin = (y_test > median).astype(float)

    configs = [
        {
            "label": "1. DecisionTree  + MSE     + Vanilla",
            "tree_cls": DecisionTree,
            "loss": MSE(),
            "boosting": VanillaBoosting(),
            "binary": False,
        },
        {
            "label": "2. ObliviousTree + MSE     + Vanilla",
            "tree_cls": ObliviousTree,
            "loss": MSE(),
            "boosting": VanillaBoosting(),
            "binary": False,
        },
        {
            "label": "3. ObliviousTree + MSE     + Ordered",
            "tree_cls": ObliviousTree,
            "loss": MSE(),
            "boosting": OrderedBoosting(),
            "binary": False,
        },
        {
            "label": "4. ObliviousTree + LogLoss + Ordered",
            "tree_cls": ObliviousTree,
            "loss": LogLoss(),
            "boosting": OrderedBoosting(),
            "binary": True,
        },
        {
            "label": "5. ObliviousTree + MSE     + SGLB",
            "tree_cls": ObliviousTree,
            "loss": MSE(),
            "boosting": SGLB(alpha=0.01, beta=10.0),
            "binary": False,
        },
        {
            "label": "6. ObliviousTree + MSE     + SGLBEnsemble",
            "tree_cls": ObliviousTree,
            "loss": MSE(),
            "boosting": SGLBEnsemble(alpha=0.01, beta=10.0, n_chains=10),
            "binary": False,
        },
    ]

    print(f"{'Config':<45}  {'Train MSE':>10}  {'Test MSE':>10}")
    print("-" * 70)

    for cfg in configs:
        if cfg["binary"]:
            Xtr, ytr = X_train, y_train_bin
            Xte, yte = X_test, y_test_bin
        else:
            Xtr, ytr = X_train, y_train
            Xte, yte = X_test, y_test

        model = GradientBoosting(
            tree_cls=cfg["tree_cls"],
            loss=cfg["loss"],
            boosting=cfg["boosting"],
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
        )
        model.fit(Xtr, ytr)

        train_pred = model.predict(Xtr)
        test_pred = model.predict(Xte)
        if cfg["binary"]:
            train_pred = 1.0 / (1.0 + np.exp(-train_pred))
            test_pred = 1.0 / (1.0 + np.exp(-test_pred))
        train_mse = mse(ytr, train_pred)
        test_mse = mse(yte, test_pred)

        label = cfg["label"]
        print(f"{label:<45}  {train_mse:>10.4f}  {test_mse:>10.4f}")

        if isinstance(cfg["boosting"], SGLBEnsemble):
            unc_train = model.predict_uncertainty(Xtr).mean()
            unc_test = model.predict_uncertainty(Xte).mean()
            print(f"  {'Epistemic uncertainty:':<43}  {unc_train:>10.5f}  {unc_test:>10.5f}")

    print()
    print(f"Irreducible noise floor (sigma^2 = 0.09): 0.0900")
    print(f"Baseline (predict train mean): {mse(y_test, np.full(len(y_test), y_train.mean())):.4f}")


if __name__ == "__main__":
    run_experiment()
