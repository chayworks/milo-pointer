"""Exploratory router analyses on the fresh steps (paper Fig. 2 and Table 5): cascade accuracy vs on-device share for several confidence
scores, and the pointer-alone comparisons. Uses the registered Opus/Sonnet outputs; no new model calls.
Usage: .venv/bin/python bench/ac_router_curves.py"""

import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = json.load(open(ROOT / "training/ac/pointer_v2/dump.json"))
TY = D["types"]
strict = [r["strict"] for r in json.load(open(ROOT / "training/ac/pointer_v2/local_eval.json"))]
idx = json.load(open(ROOT / "bench/ac_fresh1000.json"))
F = {
    m: [
        r["ok"]
        for r in json.load(open(ROOT / f"bench/results/acfs_claude-{m}-5-5_ac_fresh1000.json"))["rows"]
    ]
    for m in ("opus", "sonnet")
}
ptr = [bool(strict[i]) for i in idx]


def scores(r):
    p = r["pt"]
    ps = sorted(p, reverse=True)
    t = TY[p.index(ps[0])]
    click = t in ("click", "long_press") and r["top"]
    e1 = r["top"][0][1] if click else 1.0
    e2 = r["top"][1][1] if click and len(r["top"]) > 1 else 0.0
    ent = -sum(x * math.log(x + 1e-12) for x in p)
    return {
        "maxprob (registered)": ps[0] * e1,
        "type maxprob only": ps[0],
        "margin": (ps[0] - ps[1]) * (e1 - e2),
        "neg. entropy": -ent + math.log(e1 + 1e-12),
    }


def mcn(x, y):
    return min(1, 2 * sum(math.comb(x + y, k) for k in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0


def paired(a, b):
    n = len(a)
    x = sum(p and not q for p, q in zip(a, b))
    y = sum(q and not p for p, q in zip(a, b))
    d = (x - y) / n
    se = math.sqrt((x + y) / n - d * d) / math.sqrt(n)
    return f"{100*d:+.1f} [{100*(d-1.96*se):+.1f}, {100*(d+1.96*se):+.1f}] p={mcn(x, y):.2g}"


if __name__ == "__main__":
    S = [scores(D["res"][i]) for i in idx]
    names = list(S[0])
    curves = {}
    for m in F:
        for nm in names:
            order = sorted(range(1000), key=lambda k: -S[k][nm])
            pts = []
            for share in range(0, 101, 5):
                loc = set(order[: share * 10])
                pts.append((share, sum(ptr[k] if k in loc else F[m][k] for k in range(1000))))
            curves[f"{m}|{nm}"] = pts
    for nm in names:
        print(
            f"{nm:22}",
            "  ".join(
                f"{m}: " + " ".join(f"{s}%={v}" for s, v in curves[f"{m}|{nm}"] if s in (50, 60, 70, 80, 90))
                for m in F
            ),
        )
    print("pointer alone vs Opus-FS", paired(ptr, F["opus"]), "| vs Sonnet-FS", paired(ptr, F["sonnet"]))
    json.dump(curves, open(ROOT / "bench/results/ac_router_curves.json", "w"))
