"""Figure 1: correct steps out of the same 1,000 AndroidControl test steps (strict metric), recomputed from committed per-step results.
Usage: training/plotenv/bin/python paper/fig_ac.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent; B = R / "bench/results"
strict = [r["strict"] for r in json.load(open(R / "training/ac/pointer_v2/local_eval.json"))]
idx = json.load(open(R / "bench/ac_sample1000.json"))
def ok(f): return sum(r["ok"] for r in json.load(open(B / f))["rows"])
rows = [("Haiku 5.5 + 16 examples", ok("acfs_claude-haiku-5-5_ac_sample1000.json"), "fs"),
        ("Opus 5.5, no examples (low effort)", ok("ac_claude-opus-5-5_ac_sample1000.json"), "zs"),
        ("Sonnet 5.5, no examples (low effort)", ok("ac_claude-sonnet-5-5_ac_sample1000.json"), "zs"),
        ("Opus 5.5, no examples (medium effort)", ok("acfs_claude-opus-5-5_ac_sample1000_ZSmed.json"), "zs"),
        ("Fable 5.1 + 16 examples", ok("acfs_claude-fable-5-1_ac_sample1000.json"), "fs"),
        ("Small model, 149M", sum(strict[i] for i in idx), "ours"),
        ("Opus 5.5 + 16 examples (medium effort)", ok("acfs_claude-opus-5-5_ac_sample1000.json"), "fs")]
col = {"zs": "#999", "fs": "#2c6fbb", "ours": "#c0392b"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for k, (name, v, c) in enumerate(rows):
    ax.plot([780, v], [k, k], color="#ddd", lw=1.5, zorder=1); ax.scatter(v, k, color=col[c], s=24, zorder=2)
    ax.annotate(str(v), (v, k), xytext=(5, -3), textcoords="offset points", fontsize=7)
ax.axvline(rows[5][1], color=col["ours"], lw=0.6, ls=":")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
ax.set_xlim(780, 910); ax.set_xlabel("correct of 1,000 steps")
fig.tight_layout(); fig.savefig(R / "paper/fig_ac.pdf")
print([(r[0], r[1]) for r in rows])
