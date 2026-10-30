"""AC-Shortlist, local preparation (no frontier calls). Fresh data: the clean retrain held out every TRAIN episode with ep % 20 == 0
as validation. Half of those episodes (ep % 40 == 0) set the routing threshold; a random sample of steps from the other half
(ep % 40 == 20), which no model, threshold or prompt has seen, is the registered test sample. The clean pruned-input model
(training/kaggle_pointer-clean/pointer_clean_r3.py) gives, per sample step, its prediction, confidence and top-5 candidate elements (original
element indices, so they match the frontier model's screen numbering).
Writes bench/results/sl_local.json (validation-row index k, local decision and top-5 per sample step).
Usage (repo root): HF_HUB_OFFLINE=1 python bench/sl_prepare.py [n=1000] [seed=20261029]"""
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "release/milo-pointer"), str(ROOT / "bench")]
from ac_features import run  # noqa: E402
from ac_frontier import DATA, gold_box  # noqa: E402
from ac_topk import ranks  # noqa: E402
from milo_pointer import BASE, MiloPointer, _Net  # noqa: E402

LAYOUT = {"FrameLayout", "LinearLayout", "RelativeLayout", "ViewGroup", "ConstraintLayout", "CoordinatorLayout", "View",
          "ScrollView", "HorizontalScrollView", "RecyclerView", "ListView", "ViewPager", "DrawerLayout", "LinearLayoutCompat",
          "CardView"}  # identical to training/kaggle_pointer-clean/pointer_clean_r3.py


def keep_r3(e):
    empty = not (e["text"] or e["desc"] or e["hint"])
    inert = not any(e.get(k) for k in ("click", "long", "edit", "scroll", "checked"))
    return not (e["cls"] in LAYOUT and empty and inert)


def val_rows():
    """TRAIN rows of validation episodes (ep % 20 == 0) in file order, with the same history string as training."""
    prev, out = None, []

    def sig(r):
        return Counter((e["text"] or e["desc"] or e["rid"], e["cls"]) for e in r["els"])

    for line in gzip.open(DATA, "rt"):
        r = json.loads(line)
        s = sig(r)
        if prev and prev[0] == r["ep"] and prev[1] == r["instr"].strip():
            j = sum((s & prev[3]).values()) / max(sum((s | prev[3]).values()), 1)
            r["hist"] = f"repeat of last step (last action {prev[2]}), screen {int(round(j * 10)) * 10}% unchanged"
        else:
            r["hist"] = "new step" + (f" (last action {prev[2]})" if prev and prev[0] == r["ep"] else "")
        prev = (r["ep"], r["instr"].strip(), r["action"]["action_type"], s)
        if r["split"] == "train" and r["ep"] % 20 == 0:
            out.append(r)
    return out


def load_clean():
    m = MiloPointer.load(ROOT / "release/milo-pointer")  # tokenizer, config and device handling
    net = _Net(m.net.enc.config)
    sd = torch.load(ROOT / "training/ac/pointer_clean_r3/pointer_clean_r3.pt", map_location="cpu")
    net.load_state_dict(sd, strict=True)
    m.net = net.to(m.device).eval()
    if m.device != "cpu":
        m.net = m.net.half()
    return m


def pruned(r):
    """Prune like training, then keep original indices so candidates match the full-screen numbering."""
    keep = [k for k, e in enumerate(r["els"]) if keep_r3(e)]
    return keep


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261029
    V = val_rows()
    stored = json.load(open(ROOT / "training/ac/pointer_clean_r3/clean_r3_results.json"))["val"]
    assert len(stored) == len(V) and all(a["ep"] == b["ep"] for a, b in zip(stored, V)), "validation order mismatch"
    cal = [k for k, r in enumerate(V) if r["ep"] % 40 == 0]
    tst = [k for k, r in enumerate(V) if r["ep"] % 40 == 20]
    tau = float(np.quantile([stored[k]["conf"] for k in cal], 0.30))
    print(f"validation steps {len(V)}: calibration half {len(cal)}, test half {len(tst)}; tau = {tau:.4f}", flush=True)
    m = load_clean()
    # parity: our inference must reproduce the Kaggle run's stored confidences on the calibration half
    diffs = []
    for k in cal[:60]:
        f = run(m, V[k], pruned(V[k]))
        diffs.append(abs(f["conf"] - stored[k]["conf"]))
        assert f["ok"] == stored[k]["ok"] or abs(f["conf"] - stored[k]["conf"]) < 0.02, (k, f, stored[k])
    print(f"parity on 60 calibration steps: max |conf diff| {max(diffs):.4f}", flush=True)
    pick = sorted(random.Random(seed).sample(tst, min(n, len(tst))))
    local = []
    for j, k in enumerate(pick):
        r = V[k]
        keep = pruned(r)
        f = run(m, r, keep)
        conf2, order = ranks(m, {**r, "els": [r["els"][i] for i in keep]})
        top5 = [keep[i] for i in order[:5]]
        gb = gold_box(r)
        local.append({"k": k, "ep": r["ep"], "gt": r["action"]["action_type"], "ok": f["ok"], "conf": f["conf"],
                      "local": f["conf"] >= tau, "top5": top5,
                      "gold_in_top5": None if not gb else any(
                          (lambda e: gb[0] <= (e["b"][0] + e["b"][2]) / 2 <= gb[2] and gb[1] <= (e["b"][1] + e["b"][3]) / 2 <= gb[3])(
                              r["els"][i]) for i in top5)})
        if j % 200 == 0:
            print(j, flush=True)
    json.dump({"seed": seed, "tau": tau, "rows": local}, open(ROOT / "bench/results/sl_local.json", "w"), indent=0)
    L = local
    d = [x for x in L if not x["local"]]
    taps = [x for x in d if x["gold_in_top5"] is not None]
    print(f"sample {len(L)}: small model correct {sum(x['ok'] for x in L)}, local {sum(x['local'] for x in L)} "
          f"(correct when local {sum(x['ok'] for x in L if x['local'])}); deferred {len(d)}, of which taps {len(taps)}, "
          f"target in top 5: {sum(x['gold_in_top5'] for x in taps)}")
