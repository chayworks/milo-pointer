"""Score AC-LearnedRouter (paper/PREREGISTRATION.md) on bench/ac_confirm650.json with the frozen rule.
Usage: python bench/ac_confirm_score.py"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from ac_learned_router_freeze import score  # noqa: E402
from ac_router_score import THR, mcnemar, strict  # noqa: E402

idx = json.load(open(ROOT / "bench/ac_confirm650.json"))
rows = json.load(open(ROOT / "bench/results/ac_features_ac_confirm650.json"))["ac_confirm650"]
assert [r["i"] for r in rows] == idx
d = json.load(open(ROOT / "bench/results/acfs_claude-opus-5-5_ac_confirm650.json"))
assert [r["i"] for r in d["rows"]] == idx
opus = np.array([r["ok"] for r in d["rows"]], bool)
small = np.array([r["full"]["ok"] for r in rows], bool)
assert all(bool(strict[i]) == s for i, s in zip(idx, small)), "features must reproduce the frozen strict outcomes"
frozen = json.load(open(ROOT / "bench/results/learned_router_frozen.json"))
learned = score(frozen, rows) >= frozen["threshold"]
registered = np.array([r["full"]["conf"] for r in rows]) >= THR
n = len(idx)


def cmp(a, b):
    x, y = int((a & ~b).sum()), int((b & ~a).sum())
    dd = (x - y) / n
    se = math.sqrt((x + y) / n - dd * dd) / math.sqrt(n)
    return dd, dd - 1.96 * se, dd + 1.96 * se, x, y, mcnemar(x, y)


for name, loc in (("learned rule", learned), ("registered rule", registered)):
    cas = np.where(loc, small, opus)
    dd, lo, hi, x, y, p = cmp(cas, opus)
    print(f"{name:16} local {100 * loc.mean():.1f}%  cascade {cas.sum()}/{n}  Opus alone {opus.sum()}  small alone {small.sum()}  "
          f"diff {100 * dd:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] (+{x}/-{y}) p={p:.3g}")
    if name == "learned rule":
        print("H-LR1 (better than Opus alone):", "SUPPORTED" if x > y and p < 0.05 else "NOT supported",
              "| H-LR2 (non-inferior, margin 2):", "SUPPORTED" if lo > -0.02 else "NOT supported")
        L = cas
dd, lo, hi, x, y, p = cmp(L, np.where(registered, small, opus))
print(f"learned vs registered rule: {100 * dd:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] p={p:.3g}")
print(f"Opus cost ${d['est_cost_usd']:.2f} for {n} steps")
