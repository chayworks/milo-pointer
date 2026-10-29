"""Top-K element recall of the small model on tap steps, split by whether the registered rule keeps or defers the step, plus
how often the target is in the local top-K on the deferred taps that few-shot Opus gets wrong (the room a candidate-guided
fallback could use). Full input, no frontier calls.
Usage (repo root): HF_HUB_OFFLINE=1 .venv/bin/python bench/ac_topk.py"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "release/milo-pointer"), str(ROOT / "bench")]
from ac_features import CAP, MAXLEN, centre_in  # noqa: E402
from ac_frontier import gold_box  # noqa: E402
from milo_pointer import TYPES, MiloPointer, el_text  # noqa: E402
from parity_check import wanted_test_rows  # noqa: E402

THR, KS = 0.9854, (1, 3, 5, 10)


def ranks(m, r):
    """Confidence and the elements ordered by pointer probability (original indices)."""
    tok = m.tok
    ids = [tok.cls_token_id] + tok(f"{r['instr']} || {r['hist']} || goal: {r['goal']}", add_special_tokens=False)["input_ids"]
    ids.append(tok.sep_token_id)
    marks, shown = [], []
    for k in range(len(r["els"]))[:CAP]:
        t = tok(el_text(r["els"][k], r["w"], r["h"]), add_special_tokens=False)["input_ids"][:28]
        if len(ids) + len(t) + 2 > MAXLEN:
            break
        marks.append(len(ids))
        ids += [tok.mask_token_id] + t
        shown.append(k)
    ids.append(tok.sep_token_id)
    with torch.no_grad():
        lt, lp, _ = (x.float().cpu() for x in m.net(torch.tensor([ids], device=m.device)))
    pt = torch.softmax(lt, -1)
    ty = TYPES[int(pt.argmax())]
    s = torch.softmax(lp[marks], -1)
    conf = float(pt.max()) * (float(s.max()) if ty in ("click", "long_press") else 1.0)
    return conf, [shown[j] for j in s.argsort(descending=True).tolist()]


if __name__ == "__main__":
    m = MiloPointer.load(ROOT / "release/milo-pointer")
    out = {}
    for name in ("ac_fresh1000", "ac_confirm650"):
        idx = json.load(open(ROOT / f"bench/{name}.json"))
        opus = json.load(open(ROOT / f"bench/results/acfs_claude-opus-5-5_{name}.json"))["rows"]
        T = wanted_test_rows(set(idx))
        rows = []
        for n, i in enumerate(idx):
            r = T[i]
            gb = gold_box(r)
            if r["action"]["action_type"] not in ("click", "long_press") or gb is None:
                continue
            conf, order = ranks(m, r)
            hit = next((k for k, e in enumerate(order) if centre_in(r["els"][e], gb)), None)
            rows.append({"i": i, "conf": conf, "rank": hit, "opus_ok": bool(opus[n]["ok"])})
        del T
        out[name] = rows
        for part, sel in (("kept", lambda x: x["conf"] >= THR), ("deferred", lambda x: x["conf"] < THR)):
            R = [x for x in rows if sel(x)]
            rec = {K: sum(x["rank"] is not None and x["rank"] < K for x in R) for K in KS}
            print(name, part, len(R), "taps; top-K recall", {K: f"{v}/{len(R)}" for K, v in rec.items()},
                  "opus", sum(x["opus_ok"] for x in R))
        W = [x for x in rows if x["conf"] < THR and not x["opus_ok"]]
        print(name, "deferred taps Opus misses:", len(W), "target in local top-5:",
              sum(x["rank"] is not None and x["rank"] < 5 for x in W), flush=True)
    json.dump(out, open(ROOT / "bench/results/ac_topk.json", "w"))
