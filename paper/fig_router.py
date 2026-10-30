"""Figure 2: (a) correct steps vs API cost per 1,000 steps; (b) routed accuracy vs share of steps kept on the device, registered and
learned rules, with the perfect-router line (1,000 held-out AndroidControl steps, run 1).
Usage: training/plotenv/bin/python paper/fig_router.py"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "bench"))
from ac_learned_router_freeze import score  # noqa: E402
from ac_router_score import strict  # noqa: E402

C = json.load(open(R / "bench/results/ac_router_curves.json"))
idx = json.load(open(R / "bench/ac_fresh1000.json"))
fz = json.load(open(R / "bench/results/learned_router_frozen.json"))
F = json.load(open(R / "bench/results/ac_features.json"))["ac_fresh1000"]
small = np.array([bool(strict[t]) for t in idx])
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.5))

# (a) cost vs accuracy; small model alone costs nothing, drawn at the left edge of the log axis
pts = [("Haiku alone", 0.26, 805, "#7f8c8d"), ("Haiku, routed", 0.08, 865, "#7f8c8d"),
       ("Sonnet alone", 4.80, 857, "#c0392b"), ("Sonnet, routed", 1.41, 881, "#c0392b"),
       ("Opus alone", 9.54, 882, "#2c6fbb"), ("Opus, routed", 2.82, 892, "#2c6fbb")]
for name, x, y, col in pts:
    routed = "routed" in name
    a.scatter([x], [y], s=26, color=col, marker="o" if routed else "s", facecolors=col if routed else "none", zorder=3)
    off = {"Haiku alone": (4, -9), "Haiku, routed": (4, -9), "Sonnet alone": (-12, -10), "Sonnet, routed": (-10, 5),
           "Opus alone": (-28, -11), "Opus, routed": (-14, 5)}[name]
    a.annotate(name, (x, y), xytext=off, textcoords="offset points", fontsize=6.5, color=col)
for (_, x1, y1, col), (_, x2, y2, _) in ((pts[1], pts[0]), (pts[3], pts[2]), (pts[5], pts[4])):
    a.plot([x1, x2], [y1, y2], color=col, lw=0.6, ls=":")
a.axhline(869, color="#777", lw=0.7, ls="-.")
a.annotate("small model alone (no API cost)", (0.055, 871), fontsize=6.5, color="#777")
a.set_xscale("log"); a.set_xlim(0.05, 15); a.set_ylim(795, 900)
a.set_xticks([0.1, 0.3, 1, 3, 10]); a.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
a.set_xlabel("API cost per 1,000 steps (USD, Batch)"); a.set_ylabel("correct of 1,000"); a.set_title("(a) cost and accuracy", fontsize=8)

# (b) routing curves
opus = {r["i"]: r["ok"] for r in json.load(open(R / "bench/results/acfs_claude-opus-5-5_ac_fresh1000.json"))["rows"]}
op = np.array([opus[t] for t in idx])
order = np.argsort(-score(fz, F))
xs, ys = [], []
for k in range(0, 1001, 10):
    loc = np.zeros(1000, bool); loc[order[:k]] = True
    xs.append(k / 10); ys.append(int(np.where(loc, small, op).sum()))
reg = C["opus|maxprob (registered)"]
b.plot([p[0] for p in reg], [p[1] for p in reg], color="#2c6fbb", lw=1.4, label="registered rule")
b.plot(xs, ys, color="#2c6fbb", lw=1.0, ls="--", label="learned rule")
b.axhline(int((small | op).sum()), color="#2c6fbb", lw=0.7, ls=(0, (1, 2)))
b.annotate(f"perfect router ({int((small | op).sum())})", (2, int((small | op).sum()) - 4), fontsize=6.5, color="#2c6fbb")
b.axhline(882, color="#2c6fbb", lw=0.7, ls=":"); b.annotate("Opus alone", (2, 884), fontsize=6.5, color="#2c6fbb")
b.axhline(869, color="#777", lw=0.7, ls="-."); b.annotate("small model alone", (2, 864), fontsize=6.5, color="#777")
b.axvline(70.7, color="#555", lw=0.6, ls="--")
b.set_xlabel("steps answered on the device (%)"); b.set_ylim(845, 940); b.set_xlim(0, 100)
b.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86)); b.set_title("(b) routing to Opus 5.5", fontsize=8)
fig.tight_layout(); fig.savefig(R / "paper/fig_router.pdf")
print("learned at 70.7%:", ys[71], "oracle", int((small | op).sum()))
