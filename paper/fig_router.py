"""Figure 2: cascade accuracy vs share of steps kept on the device (1,000 fresh AndroidControl steps, run 1), registered confidence score.
Usage: training/plotenv/bin/python paper/fig_router.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent
C = json.load(open(R / "bench/results/ac_router_curves.json"))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for m, col, alone in (("opus", "#2c6fbb", 882), ("sonnet", "#c0392b", 857)):
    pts = C[f"{m}|maxprob (registered)"]; ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.4,
                                                  label=f"small model, then {m.capitalize()} 5.5")
    ax.axhline(alone, color=col, lw=0.7, ls=":")
    ax.annotate(f"{m.capitalize()} alone", (30, alone + 1.2) if m == "opus" else (80, alone - 3.4), color=col, fontsize=6.5)
ax.axvline(70.7, color="#555", lw=0.6, ls="--"); ax.annotate("registered\nthreshold", (51, 893), fontsize=6.5, color="#555")
ax.axhline(869, color="#777", lw=0.7, ls="-.")
ax.annotate("small model alone", (1, 869 - 3.2), color="#777", fontsize=6.5)
ax.set_xlabel("steps answered on the device (%)"); ax.set_ylabel("correct of 1,000"); ax.set_ylim(840, 900); ax.set_xlim(0, 100)
ax.legend(fontsize=6.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(R / "paper/fig_router.pdf"); print("ok")
