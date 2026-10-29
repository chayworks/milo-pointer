"""AC-Router (paper/PREREGISTRATION.md): frozen 149M pointer answers locally when confident, else retrieved few-shot Opus 5.5 (Opus-FS).
Rule fixed before the fresh run: confidence = max action-type probability (times the top element probability for click/long_press),
from training/ac/pointer_v2/dump.json; local if confidence >= THR, the 30th-percentile confidence on bench/ac_sample1000.json.
Usage: [MODEL=claude-sonnet-5-5] python bench/ac_router_score.py bench/ac_fresh1000.json   (default MODEL claude-opus-5-5)
"""

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THR = 0.9854
D = json.load(open(ROOT / "training/ac/pointer_v2/dump.json"))
TY = D["types"]
strict = [r["strict"] for r in json.load(open(ROOT / "training/ac/pointer_v2/local_eval.json"))]


def conf(r):
    c = max(r["pt"])
    if TY[r["pt"].index(c)] in ("click", "long_press") and r["top"]:
        c *= r["top"][0][1]
    return c


def mcnemar(x, y):
    return min(1, 2 * sum(math.comb(x + y, k) for k in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0


if __name__ == "__main__":
    sample = Path(sys.argv[1]).stem
    idx = json.load(open(ROOT / sys.argv[1]))
    M = os.environ.get("MODEL", "claude-opus-5-5")
    d = json.load(open(ROOT / f"bench/results/acfs_{M}_{sample}.json"))
    rows = d["rows"]
    assert [r["i"] for r in rows] == idx
    loc = [conf(D["res"][i]) >= THR for i in idx]
    ptr = [bool(strict[i]) for i in idx]
    op = [bool(r["ok"]) for r in rows]
    cas = [p if l else o for p, o, l in zip(ptr, op, loc)]
    n = len(idx)
    x = sum(c and not o for c, o in zip(cas, op))
    y = sum(o and not c for c, o in zip(cas, op))
    diff = (x - y) / n
    se = math.sqrt((x + y) / n - diff**2) / math.sqrt(n)
    print(
        f"{M}: n={n}  pointer {sum(ptr)}  FS {sum(op)}  router cascade {sum(cas)}  local share {100 * sum(loc) / n:.1f}%"
    )
    print(
        f"cascade - {M} alone: {100 * diff:+.1f} pts [{100 * (diff - 1.96 * se):+.1f}, {100 * (diff + 1.96 * se):+.1f}] (+{x}/-{y}) McNemar p={mcnemar(x, y):.3g}"
    )
    xp = sum(c and not p for c, p in zip(cas, ptr))
    yp = sum(p and not c for c, p in zip(cas, ptr))
    print(f"cascade - pointer: {100 * (xp - yp) / n:+.1f} pts (+{xp}/-{yp}) p={mcnemar(xp, yp):.3g}")
    print(
        f"pointer accuracy on its local steps {sum(p for p, l in zip(ptr, loc) if l)}/{sum(loc)}; {M} on the same {sum(o for o, l in zip(op, loc) if l)}"
    )
    c1k = 1000 * d["est_cost_usd"] / n
    print(
        f"{M} ${c1k:.2f}/1k steps (Batch); cascade about ${c1k * (1 - sum(loc) / n):.2f}/1k (measured: bench/results/*_usage.json)"
    )
    print(
        f"non-inferior to {M} alone (margin 2):",
        "SUPPORTED" if diff - 1.96 * se > -0.02 else "NOT supported",
        f"| better than {M} alone:",
        "SUPPORTED" if x > y and mcnemar(x, y) < 0.05 else "NOT supported",
        "| >=60% local:",
        "SUPPORTED" if sum(loc) / n >= 0.6 else "NOT supported",
    )
