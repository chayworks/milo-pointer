"""The paper's headline numbers, recomputed from the released per-step outputs (run: pytest -q)."""
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from ac_router_score import THR, D, conf, strict  # noqa: E402

IDX = json.load(open(ROOT / "bench/ac_fresh1000.json"))
LOCAL = [conf(D["res"][i]) >= THR for i in IDX]
SMALL = [bool(strict[i]) for i in IDX]


def outcomes(model, run=""):
    rows = json.load(open(ROOT / f"bench/results/acfs_{model}_ac_fresh1000{run}.json"))["rows"]
    assert [r["i"] for r in rows] == IDX
    return [bool(r["ok"]) for r in rows]


def routed(frontier):
    return [s if l else f for s, f, l in zip(SMALL, frontier, LOCAL)]


def lower_bound(a, b):
    n = len(a); x = sum(p and not q for p, q in zip(a, b)); y = sum(q and not p for p, q in zip(a, b))
    d = (x - y) / n
    return d - 1.96 * math.sqrt((x + y) / n - d * d) / math.sqrt(n)


def test_threshold_and_local_share():
    assert THR == 0.9854 and sum(LOCAL) == 707


def test_small_model_alone():
    assert sum(SMALL) == 869


def test_routing_to_opus_is_non_inferior_in_both_runs():
    for run, alone, cascade in (("", 882, 892), ("_R2", 879, 890)):
        opus = outcomes("claude-opus-5-5", run)
        assert (sum(opus), sum(routed(opus))) == (alone, cascade)
        assert lower_bound(routed(opus), opus) > -0.02


def test_routing_to_sonnet_reaches_opus():
    sonnet, opus = outcomes("claude-sonnet-5-5"), outcomes("claude-opus-5-5")
    assert (sum(sonnet), sum(routed(sonnet))) == (857, 881)
    assert lower_bound(routed(sonnet), opus) > -0.02


def test_routing_to_haiku_does_not_reach_opus():
    haiku, opus = outcomes("claude-haiku-5-5"), outcomes("claude-opus-5-5")
    assert (sum(haiku), sum(routed(haiku))) == (805, 865)
    assert lower_bound(routed(haiku), opus) < -0.02


def test_reproduce_script_runs():
    subprocess.run(["bash", str(ROOT / "reproduce.sh")], check=True, capture_output=True)


def test_confirmatory_learned_router_beats_opus():
    """AC-LearnedRouter: frozen learned rule on the last 650 unused test steps."""
    out = subprocess.run([sys.executable, str(ROOT / "bench/ac_confirm_score.py")], check=True, capture_output=True, text=True).stdout
    assert "cascade 599/650  Opus alone 581" in out and "H-LR1 (better than Opus alone): SUPPORTED" in out
