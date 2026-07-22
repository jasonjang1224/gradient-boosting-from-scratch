# Gradient Boosting from Scratch

This project derives and implements a modern Gradient Boosting Machine (GBM) from the ground up, assuming only undergraduate calculus (derivatives, chain rule, Taylor expansion). Every equation is derived step by step, every component is implemented in pure NumPy, and every line of code is mapped back to the math that produced it.

The goal of this document and codebase is to be **easier to follow, yet more rigorous, than any material currently available**. "Easier" doesn't mean skipping steps — it means no step is skipped, so the derivation can be followed all the way through without unexplained jumps. "Rigorous" means every equation in the writeup is explicitly traced to the exact line of code that implements it.

## What it covers

Starting from Vanilla GBM, five improvements are introduced in sequence, each motivated by a concrete limitation of the previous one.

1. **Newton Step** — using the second derivative (Hessian) in addition to the first-order gradient to compute the optimal leaf value
2. **Oblivious Trees** — symmetric trees where every node at the same depth shares the same split rule (CatBoost's approach)
3. **Ordered Target Statistics** — encoding categorical features using a random permutation to prevent target leakage
4. **Ordered Boosting** — removing the prediction shift caused by a tree's own predictions being used in its own training, via an ordering-based technique
5. **Posterior Sampling (SGLB)** — Stochastic Gradient Langevin Boosting, which injects noise into the gradient to sample from a Bayesian posterior, and ensembles multiple chains to estimate epistemic uncertainty

## Repository structure

```
src/
  vanilla_gbm.py   Baseline CART-style decision tree + Vanilla GBM (first-order gradient only)
  tree.py          Oblivious Tree — Newton-step leaf values, shared splits per depth level
  loss.py          Loss functions: MSE, LogLoss (gradient / hessian / initial prediction)
  boosting.py      Boosting strategies: VanillaBoosting, OrderedBoosting, SGLB, SGLBEnsemble
  encoder.py       OrderedTargetEncoder — ordered target-statistic encoding
  experiment.py    Reproduces a 6-configuration comparison experiment

gradient-boosting-from-scratch.pdf   Full derivation and write-up
```

## Running it

```bash
pip install numpy
python src/experiment.py
```

`experiment.py` trains and evaluates the following 6 configurations on synthetic data of the form `y = x0^2 + 0.5*x1 + noise`, comparing train/test MSE.

1. DecisionTree + MSE + Vanilla
2. ObliviousTree + MSE + Vanilla
3. ObliviousTree + MSE + Ordered Boosting
4. ObliviousTree + LogLoss + Ordered Boosting (binary classification)
5. ObliviousTree + MSE + SGLB
6. ObliviousTree + MSE + SGLBEnsemble (10-chain ensemble, includes uncertainty estimation)

## Documentation

`gradient-boosting-from-scratch.pdf` contains the full derivation, the motivation behind each improvement, and its mapping to the corresponding code. Every component is implemented from scratch in pure NumPy, with no dependency on external ML libraries (scikit-learn, XGBoost, CatBoost, etc.).
