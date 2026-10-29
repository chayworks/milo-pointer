"""Learned deferral router (exploratory). Trained only on ac_sample1000 (the sample used to set the registered threshold, where Opus-FS
outputs also exist); tested on ac_fresh1000 against the registered max-probability rule at matched local shares.
Model A: logistic regression predicting whether the small model is right. Model B: also predicts whether Opus is right, and keeps a step
local when P(small right) - P(Opus right) is highest. Usage: .venv/bin/python bench/ac_learned_router.py"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F = json.load(open(ROOT / "bench/results/ac_features.json"))
GROUPS = {"click": 0, "long_press": 0, "input_text": 1, "open_app": 2, "wait": 3, "navigate_back": 4, "navigate_home": 4}


def feats(r):
    f = r["full"]
    g = np.zeros(6)
    g[GROUPS.get(f["type"], 5)] = 1   # 5 = any scroll
    return np.concatenate([[math.log(f["p_type"] + 1e-9), f["type_margin"], f["type_entropy"], math.log(f["p_el"] + 1e-9), f["el_margin"],
                            math.log(f["p_span"] + 1e-9), f["icon"], math.log(f["len"]), math.log(r["n_els"] + 1)], g])


def fit(X, y, l2=1.0, it=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y))
    return lambda X2: 1 / (1 + np.exp(-np.c_[np.ones(len(X2)), (X2 - mu) / sd] @ w))


def outcomes(model, sample, run=""):
    return np.array([r["ok"] for r in json.load(open(ROOT / f"bench/results/acfs_{model}_{sample}{run}.json"))["rows"]], dtype=bool)


def cascade(order, small, front, share):
    k = int(round(share * len(order)))
    loc = np.zeros(len(order), bool)
    loc[order[:k]] = True
    return int(np.where(loc, small, front).sum())


if __name__ == "__main__":
    tr, te = F["ac_sample1000"], F["ac_fresh1000"]
    Xtr, Xte = np.array([feats(r) for r in tr]), np.array([feats(r) for r in te])
    ytr = np.array([r["full"]["ok"] for r in tr], float)
    small_te = np.array([r["full"]["ok"] for r in te], bool)
    pA = fit(Xtr, ytr)(Xte)
    pB = pA - fit(Xtr, outcomes("claude-opus-5-5", "ac_sample1000").astype(float))(Xte)
    reg = np.array([r["full"]["conf"] for r in te])
    orders = {"registered max-prob": np.argsort(-reg), "learned P(small right)": np.argsort(-pA), "learned P(small)-P(Opus)": np.argsort(-pB)}
    arms = {"Opus r1": outcomes("claude-opus-5-5", "ac_fresh1000"), "Opus r2": outcomes("claude-opus-5-5", "ac_fresh1000", "_R2"),
            "Sonnet r1": outcomes("claude-sonnet-5-5", "ac_fresh1000"), "Sonnet r2": outcomes("claude-sonnet-5-5", "ac_fresh1000", "_R2"),
            "Haiku": outcomes("claude-haiku-5-5", "ac_fresh1000")}
    curves = {}
    for name, o in orders.items():
        for arm, front in arms.items():
            curves[f"{name}|{arm}"] = [(s, cascade(o, small_te, front, s / 100)) for s in range(0, 101, 5)]
    print("local share 70.7% (registered):", {a: {n: cascade(o, small_te, f, 0.707) for n, o in orders.items()} for a, f in arms.items()})
    for s in (50, 60, 80, 90):
        print(f"local {s}%:", {a: {n.split()[0] + ('-B' if 'Opus)' in n else ''): cascade(o, small_te, f, s / 100) for n, o in orders.items()} for a, f in list(arms.items())[:3]})
    # risk control: threshold set on the training sample so that local steps there have <= 5% errors
    pA_tr = fit(Xtr, ytr)(Xtr)
    for target in (0.03, 0.05):
        ths = sorted(pA_tr, reverse=True)
        best = max((t for t in ths if (1 - ytr[pA_tr >= t].mean()) <= target), default=None, key=lambda t: (pA_tr >= t).sum())
        loc = pA >= best
        print(f"error target {target:.0%} (set on training sample): fresh local share {loc.mean():.1%}, local error {1 - small_te[loc].mean():.1%}")
    json.dump(curves, open(ROOT / "bench/results/ac_learned_router.json", "w"))
