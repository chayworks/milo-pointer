"""Statistics for the paper: 95% bootstrap confidence intervals (resampling EPISODES, since steps within an episode are
correlated) per split, and paired bootstrap tests between models scored on the same test steps.
Inputs are per-step result files (strict metric), aligned by (episode, step order).
Usage: python3 bench/ac_stats.py name=path.json [name=path.json ...]"""
import json, random, sys
from collections import defaultdict

def load(p):
    d = json.load(open(p)); rows = d["res"] if isinstance(d, dict) and "res" in d else d
    key = defaultdict(int); out = {}
    for r in rows:
        ok = r.get("strict", r.get("ok")); out[(r["ep"], key[r["ep"]])] = bool(ok); key[r["ep"]] += 1
    return out
M = {a.split("=")[0]: load(a.split("=")[1]) for a in sys.argv[1:]}
keys = sorted(set.intersection(*[set(m) for m in M.values()])); eps = sorted({k[0] for k in keys})
by_ep = defaultdict(list)
for k in keys: by_ep[k[0]].append(k)
subs = json.load(open("bench/androidcontrol_test_subsplits.json")); subs["ALL"] = eps
R = random.Random(0); B = 2000
def acc(m, ks): return 100 * sum(M[m][k] for k in ks) / max(len(ks), 1)
for split, se in subs.items():
    se = [e for e in set(se) if e in by_ep]
    boots = []
    for _ in range(B):
        samp = [R.choice(se) for _ in se]; boots.append([k for e in samp for k in by_ep[e]])
    print(f"\n{split} ({sum(len(by_ep[e]) for e in se)} steps, {len(se)} episodes)")
    for m in M:
        vals = sorted(acc(m, ks) for ks in boots)
        print(f"  {m:10s} {acc(m, [k for e in se for k in by_ep[e]]):5.1f}  95% CI [{vals[int(.025 * B)]:.1f}, {vals[int(.975 * B)]:.1f}]")
    names = list(M)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]; diffs = sorted(acc(a, ks) - acc(b, ks) for ks in boots)
            p = 2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / B
            print(f"  {a} − {b}: {acc(a, [k for e in se for k in by_ep[e]]) - acc(b, [k for e in se for k in by_ep[e]]):+.1f}  CI [{diffs[int(.025 * B)]:+.1f}, {diffs[int(.975 * B)]:+.1f}]  p≈{max(p, 1 / B):.4f}")
