"""Score AC-Shortlist (paper/PREREGISTRATION.md) from bench/results/sl_local.json and the frontier arms' outputs.
Usage: python bench/sl_score.py [model=claude-haiku-5-5]"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-5-5"
L = json.load(open(ROOT / "bench/results/sl_local.json"))["rows"]
N = len(L)


def arm(name):
    p = ROOT / f"bench/results/sl_{M}_{name}.json"
    return {x["j"]: x["ok"] for x in json.load(open(p))["rows"]} if p.exists() else None


def routed(fb):
    return [L[j]["ok"] if L[j]["local"] else fb[j] for j in range(N)]


def cmp(a, b):
    x = sum(p and not q for p, q in zip(a, b))
    y = sum(q and not p for p, q in zip(a, b))
    d = (x - y) / N
    se = math.sqrt(max((x + y) / N - d * d, 0)) / math.sqrt(N)
    p = min(1, 2 * sum(math.comb(x + y, k) for k in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0
    return d, d - 1.96 * se, d + 1.96 * se, p, x, y


def fmt(c):
    return f"{100*c[0]:+.1f} [{100*c[1]:+.1f}, {100*c[2]:+.1f}] (+{c[4]}/-{c[5]}) p={c[3]:.3g}"


full, fdef, cf, co = arm("full"), arm("full_def"), arm("cand_full"), arm("cand_only")
base = full or fdef
assert base and cf and co, "missing arm outputs"
R_full, R_cf, R_co = routed(base), routed(cf), routed(co)
print(f"{N} steps; local {sum(x['local'] for x in L)}; small alone {sum(x['ok'] for x in L)}")
print(f"routed: full {sum(R_full)}  cand_full {sum(R_cf)}  cand_only {sum(R_co)}" + (f"  {M} alone {sum(full.values())}" if full else ""))
D = [j for j in range(N) if not L[j]["local"]]
print(f"deferred {len(D)}: full {sum(base[j] for j in D)}  cand_full {sum(cf[j] for j in D)}  cand_only {sum(co[j] for j in D)}")
T = [j for j in D if L[j]["gold_in_top5"] is not None]
print(f"deferred taps {len(T)}: full {sum(base[j] for j in T)}  cand_full {sum(cf[j] for j in T)}  cand_only {sum(co[j] for j in T)}")
tests = [("H-SL1 cand_full better than full (routed)", cmp(R_cf, R_full), "sup"),
         ("H-SL2 cand_only non-inferior to full (routed), margin 2", cmp(R_co, R_full), "ni")]
if full:
    tests.append((f"H-SL3 routed cand_full better than {M} alone", cmp(R_cf, [full[j] for j in range(N)]), "sup"))
ps = sorted(range(len(tests)), key=lambda i: tests[i][1][3])
holm = {}
for r, i in enumerate(ps):  # Holm step-down over the superiority p-values (non-inferiority uses its CI bound)
    holm[i] = tests[i][1][3] <= 0.05 / (len(tests) - r)
for i, (name, c, kind) in enumerate(tests):
    ok = c[1] > -0.02 if kind == "ni" else (c[0] > 0 and holm[i])
    print(f"{name}: {fmt(c)} -> {'SUPPORTED' if ok else 'NOT supported'}")
if not full:
    print("H-SL3: not run (deferred-step arms only)")
else:
    print(f"secondary: routed full vs {M} alone", fmt(cmp(R_full, [full[j] for j in range(N)])))
S = ROOT / "bench/results/sl_claude-sonnet-5-5_full_def.json"
if S.exists():  # H-SL4 (amendment): routed Haiku cand_full non-inferior to routed Sonnet full_def, margin 2
    sd = {x["j"]: x["ok"] for x in json.load(open(S))["rows"]}
    R_s = routed(sd)
    small = [x["ok"] for x in L]
    print(f"Sonnet full_def: deferred {sum(sd[j] for j in D)}, taps {sum(sd[j] for j in T)}; routed {sum(R_s)}")
    c = cmp(R_cf, R_s)
    print(f"H-SL4 routed {M} cand_full non-inferior to routed Sonnet full_def, margin 2: {fmt(c)} -> "
          f"{'SUPPORTED' if c[1] > -0.02 else 'NOT supported'}")
    print("secondary: routed Sonnet full_def vs small alone", fmt(cmp(R_s, small)))
    if full:
        print(f"secondary: routed Sonnet full_def vs {M} alone", fmt(cmp(R_s, [full[j] for j in range(N)])))
print("secondary: routed cand_full vs small alone", fmt(cmp(R_cf, [x["ok"] for x in L])))
print("secondary: routed cand_only vs small alone", fmt(cmp(R_co, [x["ok"] for x in L])))
if S.exists():
    print("secondary: routed cand_only vs routed Sonnet full_def", fmt(cmp(R_co, R_s)))
