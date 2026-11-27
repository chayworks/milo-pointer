"""AC-GoalFront scoring (paper/PREREGISTRATION.md): F = Sonnet 5.5 goal-only with 16 retrieved examples on every test step;
R = AC-Planner2's local action where its plan check trusts the plan (comb >= the frozen dev threshold), otherwise F's action.
Usage: python bench/goalfront/score.py   (reads bench/goalfront/prompts.jsonl, bench/goalfront/raw/*/goalfront_*_test_*.jsonl and
training/kaggle_planner2/planner2_results.json; writes bench/goalfront/goalfront_results.json)"""
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
from ac_frontier import gold_type, parse, strict_ok  # noqa: E402

gold = [json.loads(line)["gold"] for line in open(ROOT / "bench/goalfront/prompts.jsonl") if '"split": "test"' in line[:30]]
p2 = json.load(open(ROOT / "training/kaggle_planner2/planner2_results.json"))
thr, local = p2["H-P2b"]["thr"], p2["steps"]["test"]
assert len(gold) == len(local) == 1804
replies = {}
for f in sorted(glob.glob(str(ROOT / "bench/goalfront/raw/*/goalfront_*_test_*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        replies[r["i"]] = r
missing = [i for i in range(len(gold)) if i not in replies]
print(f"replies {len(replies)} / {len(gold)}; missing {len(missing)}")
if missing:
    sys.exit("incomplete: run the remaining slices first")

F, R, kept, types, cost_f, cost_r = [], [], [], [], 0, 0
unparsed = failed = 0
for i, (g, s) in enumerate(zip(gold, local)):
    rep = replies[i]
    failed += rep["reply"].startswith("ERROR")
    p = parse(rep["reply"])
    unparsed += not p.get("action_type")
    f_ok = bool(strict_ok({"action": g["action"], "els": g["els"]}, p))
    k = s["comb"] >= thr
    F.append(f_ok)
    R.append(bool(s["ok"]) if k else f_ok)
    kept.append(k)
    types.append(gold_type(g["action"]))
    c = (rep.get("usage") or {}).get("nanodollars", 0) / 1e9
    cost_f += c
    cost_r += 0 if k else c
n = len(F)
x = sum(a and not b for a, b in zip(R, F))
y = sum(b and not a for a, b in zip(R, F))
d = (x - y) / n
se = math.sqrt(max((x + y) / n - d * d, 0) / n)
lo, hi = d - 1.96 * se, d + 1.96 * se
p = min(1.0, 2 * sum(math.comb(x + y, j) for j in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0
h1 = lo > -0.03
h2 = h1 and p < 0.05 and d > 0
ks = [i for i in range(n) if kept[i]]
by_type = defaultdict(lambda: [0, 0, 0])
for t, a, b in zip(types, R, F):
    by_type[t][0] += 1
    by_type[t][1] += a
    by_type[t][2] += b
res = {
    "n": n, "acc_R": sum(R) / n, "acc_F": sum(F) / n, "diff": d, "lo": lo, "hi": hi, "R_only": x, "F_only": y, "mcnemar_p": p,
    "H-GF1_margin3": h1, "margin2_would_pass": lo > -0.02, "H-GF2": h2 if h1 else None,
    "sent_to_sonnet": 1 - len(ks) / n, "cost_F": cost_f, "cost_R": cost_r,
    "kept_local_acc": sum(local[i]["ok"] for i in ks) / len(ks), "kept_sonnet_acc": sum(F[i] for i in ks) / len(ks),
    "deferred_sonnet_acc": sum(F[i] for i in range(n) if not kept[i]) / (n - len(ks)),
    "unparsed": unparsed, "failed_calls": failed,
    "by_type": {t: {"n": v[0], "R": v[1] / v[0], "F": v[2] / v[0]} for t, v in sorted(by_type.items())},
    "steps": [{"i": i, "kept": kept[i], "R": R[i], "F": F[i], "reply": replies[i]["reply"], "usage": replies[i].get("usage")}
              for i in range(n)],
}
json.dump(res, open(ROOT / "bench/goalfront/goalfront_results.json", "w"), indent=1)
print(f"R {res['acc_R']:.3f}  F {res['acc_F']:.3f}  R-F {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  R only {x}, F only {y}, McNemar p {p:.2e}")
print(f"H-GF1 (margin 3): {'PASS' if h1 else 'FAIL'}; margin 2 would {'pass' if lo > -0.02 else 'fail'}; "
      f"H-GF2: {'not tested' if not h1 else 'PASS' if h2 else 'FAIL'}")
print(f"sent to Sonnet {res['sent_to_sonnet']:.3f}; Sonnet cost R ${cost_r:.2f} vs F ${cost_f:.2f}")
print(f"kept steps: local {res['kept_local_acc']:.3f} vs Sonnet {res['kept_sonnet_acc']:.3f}; deferred steps: Sonnet "
      f"{res['deferred_sonnet_acc']:.3f}; unparsed {unparsed}, failed {failed}")
for t, v in res["by_type"].items():
    print(f"  {t:14} n {v['n']:4}  R {v['R']:.3f}  F {v['F']:.3f}")
