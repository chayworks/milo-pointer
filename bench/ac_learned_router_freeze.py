"""Freeze the learned deferral router for its confirmatory test: fit both logistic models on ac_sample1000 (as in
ac_learned_router.py), and fix the threshold on the score s = P(small right) - P(Opus right) at the value that keeps 70.7% of the
development steps (ac_fresh1000) local. Writes bench/results/learned_router_frozen.json.
Usage: python bench/ac_learned_router_freeze.py"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "bench"))
from ac_learned_router import F, feats, outcomes  # noqa: E402


def fit_params(X, y, l2=1.0, it=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y))
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def prob(P, X):
    Z = np.c_[np.ones(len(X)), (X - np.array(P["mu"])) / np.array(P["sd"])]
    return 1 / (1 + np.exp(-Z @ np.array(P["w"])))


def score(frozen, rows):
    X = np.array([feats(r) for r in rows])
    return prob(frozen["small"], X) - prob(frozen["opus"], X)


if __name__ == "__main__":
    tr = F["ac_sample1000"]
    Xtr = np.array([feats(r) for r in tr])
    frozen = {"small": fit_params(Xtr, np.array([r["full"]["ok"] for r in tr], float)),
              "opus": fit_params(Xtr, outcomes("claude-opus-5-5", "ac_sample1000").astype(float))}
    s = np.sort(score(frozen, F["ac_fresh1000"]))[::-1]
    frozen["threshold"] = float(s[706])   # 707 of 1,000 development steps at or above it
    json.dump(frozen, open(ROOT / "bench/results/learned_router_frozen.json", "w"), indent=1)
    print("threshold", frozen["threshold"])
