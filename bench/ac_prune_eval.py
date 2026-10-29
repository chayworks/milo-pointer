"""Accuracy, routing and input length of the small model when empty layout containers are removed from the screen (rule R3), on the
1,000 fresh steps, with the unchanged checkpoint. Usage: HF_HUB_OFFLINE=1 .venv/bin/python bench/ac_prune_eval.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "release/milo-pointer"), str(ROOT / "bench")]
from ac_features import run  # noqa: E402
from ac_router_score import THR, mcnemar  # noqa: E402
from milo_pointer import MiloPointer  # noqa: E402
from parity_check import wanted_test_rows  # noqa: E402

LAYOUT = {"FrameLayout", "LinearLayout", "RelativeLayout", "ViewGroup", "ConstraintLayout", "CoordinatorLayout", "View", "ScrollView",
          "HorizontalScrollView", "RecyclerView", "ListView", "ViewPager", "DrawerLayout", "LinearLayoutCompat", "CardView"}


def keep_r3(e):
    """Drop layout containers that show nothing and cannot be acted on."""
    empty = not (e["text"] or e["desc"] or e["hint"])
    inert = not any(e.get(k) for k in ("click", "long", "edit", "scroll", "checked"))
    return not (e["cls"] in LAYOUT and empty and inert)


if __name__ == "__main__":
    m = MiloPointer.load(ROOT / "release/milo-pointer")
    idx = json.load(open(ROOT / "bench/ac_fresh1000.json"))
    T = wanted_test_rows(set(idx))
    res = [run(m, T[i], [k for k, e in enumerate(T[i]["els"]) if keep_r3(e)]) for i in idx]
    json.dump(res, open(ROOT / "bench/results/ac_prune_r3.json", "w"))
    full = json.load(open(ROOT / "bench/results/ac_features.json"))["ac_fresh1000"]
    ok = [r["ok"] for r in res]
    L = sorted(r["len"] for r in res)
    Lf = sorted(r["full"]["len"] for r in full)
    print(f"R3: alone {sum(ok)} (full {sum(r['full']['ok'] for r in full)}); median tokens {L[500]} (full {Lf[500]}), p90 {L[900]} (full {Lf[900]})")
    loc = [r["conf"] >= THR for r in res]
    for mname in ("opus", "sonnet"):
        F = [x["ok"] for x in json.load(open(ROOT / f"bench/results/acfs_claude-{mname}-5-5_ac_fresh1000.json"))["rows"]]
        cas = [p if l else f for p, f, l in zip(ok, F, loc)]
        x = sum(c and not f for c, f in zip(cas, F))
        y = sum(f and not c for c, f in zip(cas, F))
        print(f"  R3 -> {mname}: {sum(cas)} vs {mname} alone {sum(F)} (+{x}/-{y}, p={mcnemar(x, y):.2g}); local {100 * sum(loc) / 1000:.1f}%")
