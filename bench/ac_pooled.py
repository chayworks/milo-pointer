"""Pooled paired estimate of routing vs Opus alone over all 1,650 test steps the routing rules never saw (held-out 1,000 +
confirmatory 650), on original labels; curated-label pooling is in ac_curated_learned.py. Not registered.
Usage: .venv/bin/python bench/ac_pooled.py"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from ac_learned_router_freeze import score  # noqa: E402
from ac_router_score import THR, D, conf, mcnemar, strict  # noqa: E402

fz = json.load(open(ROOT / "bench/results/learned_router_frozen.json"))
F = json.load(open(ROOT / "bench/results/ac_features.json"))
F.update(json.load(open(ROOT / "bench/results/ac_features_ac_confirm650.json")))
tot = {}
for s in ("ac_fresh1000", "ac_confirm650"):
    idx = json.load(open(ROOT / f"bench/{s}.json"))
    opus = {r["i"]: r["ok"] for r in json.load(open(ROOT / f"bench/results/acfs_claude-opus-5-5_{s}.json"))["rows"]}
    rules = {"registered": {t: conf(D["res"][t]) >= THR for t in idx},
             "learned": dict(zip(idx, score(fz, F[s]) >= fz["threshold"])), "small alone": {t: True for t in idx}}
    for name, loc in rules.items():
        cas = [bool(strict[t]) if loc[t] else opus[t] for t in idx]
        x = sum(c and not opus[t] for c, t in zip(cas, idx))
        y = sum(opus[t] and not c for c, t in zip(cas, idx))
        n = len(idx)
        print(f"{s} {name}: {100*(x-y)/n:+.1f} (+{x}/-{y}, n={n})")
        a = tot.setdefault(name, [0, 0, 0]); a[0] += x; a[1] += y; a[2] += n
for name, (x, y, n) in tot.items():
    d = (x - y) / n
    se = math.sqrt((x + y) / n - d * d) / math.sqrt(n)
    print(f"POOLED {name}: {100*d:+.1f} [{100*(d-1.96*se):+.1f}, {100*(d+1.96*se):+.1f}] (+{x}/-{y}, n={n}) p={mcnemar(x, y):.3g}")
