"""Re-score both routing rules on AndroidControl-Curated labels, on the 1,000 held-out steps and the 650 confirmatory steps, plus
the oracle ceiling (a step counts if either model is right). Same curated matching as ac_curated_score.py. Not registered.
Usage: .venv/bin/python bench/ac_curated_learned.py /path/to/android_control_low_bbox.json"""
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from ac_learned_router_freeze import score  # noqa: E402
from ac_router_score import THR, D, conf  # noqa: E402

SAMPLES = ("ac_fresh1000", "ac_confirm650")


def mcn(x, y):
    return min(1, 2 * sum(math.comb(x + y, k) for k in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0


C = {}
for x in json.load(open(sys.argv[1])):
    p = x["image"].split("/")
    C[(int(p[1]), int(p[2][5:-4]))] = x
S = {s: json.load(open(ROOT / f"bench/{s}.json")) for s in SAMPLES}
want = set().union(*S.values())
LE = json.load(open(ROOT / "training/ac/pointer_v2/local_eval.json"))
rows, boxes, t = {}, {}, -1
for line in gzip.open(ROOT / "training/ac/androidcontrol_text.jsonl.gz", "rt"):
    r = json.loads(line)
    if r["split"] != "test":
        continue
    t += 1
    c = C.get((r["ep"], r["step"]))
    if c is None or t not in want:
        continue
    rows[t] = (r["action"], c)
    boxes[t] = [e["b"] for e in r["els"][:220]]

import ac_curated_score as cs  # noqa: E402  (reuses its matching rules; its own report runs on import)

cs.rows, cs.boxes = rows, boxes
fz = json.load(open(ROOT / "bench/results/learned_router_frozen.json"))
F = json.load(open(ROOT / "bench/results/ac_features.json"))
F.update(json.load(open(ROOT / "bench/results/ac_features_ac_confirm650.json")))
print("\n--- routing rules on curated labels ---")
for s, idx in S.items():
    learn = dict(zip(idx, score(fz, F[s]) >= fz["threshold"]))
    opus = {r["i"]: r["got"] for r in json.load(open(ROOT / f"bench/results/acfs_claude-opus-5-5_{s}.json"))["rows"]}
    T = [t for t in idx if t in rows]
    P = {t: cs.ptr_ok(t) for t in T}
    O = {t: cs.claude_ok(t, opus[t]) for t in T}
    n = len(T)
    out = [f"{s}: matched {n}  small {sum(P.values())}  Opus {sum(O.values())}  oracle {sum(P[t] or O[t] for t in T)}"]
    for name, loc in (("registered", {t: conf(D["res"][t]) >= THR for t in T}), ("learned", learn)):
        cas = {t: P[t] if loc[t] else O[t] for t in T}
        x = sum(cas[t] and not O[t] for t in T)
        y = sum(O[t] and not cas[t] for t in T)
        d = (x - y) / n
        se = math.sqrt((x + y) / n - d * d) / math.sqrt(n)
        out.append(f"  {name}: routed {sum(cas.values())} ({100*sum(cas.values())/n:.1f}%) vs Opus {100*d:+.1f} "
                   f"[{100*(d-1.96*se):+.1f},{100*(d+1.96*se):+.1f}] +{x}/-{y} p={mcn(x, y):.2g}; local {np.mean([loc[t] for t in T]):.3f}")
    print("\n".join(out))
