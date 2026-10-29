"""AC-Fair scoring (paper/PREREGISTRATION.md): frozen 149M pointer (pointer_v2 strict) vs retrieved few-shot Claude on the same steps.
Paired exact McNemar (two-sided), overall and on the unseen-app subsplit (episode lists in bench/androidcontrol_test_subsplits.json).
Usage: python bench/ac_fair_score.py bench/ac_sample1000.json"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
strict = [r["strict"] for r in json.load(open(ROOT / "training/ac/pointer_v2/local_eval.json"))]
sample = Path(sys.argv[1]).stem
idx = json.load(open(ROOT / sys.argv[1]))
unseen = set(json.load(open(ROOT / "bench/androidcontrol_test_subsplits.json"))["app_unseen"])


def mcnemar(a, b):
    x = sum(p and not q for p, q in zip(a, b))
    y = sum(q and not p for p, q in zip(a, b))
    p = min(1, 2 * sum(math.comb(x + y, k) for k in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0
    return x, y, p


def report(name, f):
    if not f.exists():
        print(f"{name:12} (missing {f.name})")
        return
    d = json.load(open(f))
    rows = d["rows"]
    assert [r["i"] for r in rows] == idx[: len(rows)]
    ptr = [bool(strict[r["i"]]) for r in rows]
    cl = [bool(r["ok"]) for r in rows]
    x, y, p = mcnemar(ptr, cl)
    n = len(rows)
    u = [k for k, r in enumerate(rows) if r["ep"] in unseen]
    xu, yu, pu = mcnemar([ptr[k] for k in u], [cl[k] for k in u])
    print(
        f"{name:12} n={n} pointer {sum(ptr)} vs {sum(cl)}  (+{x}/-{y}) p={p:.2g}  | app_unseen {sum(ptr[k] for k in u)} vs "
        f"{sum(cl[k] for k in u)} of {len(u)} p={pu:.2g}  | ${1000 * d['est_cost_usd'] / n:.2f}/1k steps"
        + (f"  errored {d['errored']}" if d.get("errored") else "")
    )
    return sum(cl), sum(ptr)


R = ROOT / "bench/results"
report("Opus-0shot", R / f"ac_claude-opus-5-5_{sample}.json")
o = report("Opus-FS", R / f"acfs_claude-opus-5-5_{sample}.json")
f = report("Fable-FS", R / f"acfs_claude-fable-5-1_{sample}.json")
if o and f and len(idx) == 1000:
    print(
        "KILL CHECK:",
        (
            "FAILS - a frontier arm ties or beats the pointer"
            if max(o[0], f[0]) >= o[1]
            else "passes - pointer ahead of both"
        ),
    )
