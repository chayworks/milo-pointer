"""Run the released small model over both 1,000-step samples, with the full screen and with empty layout wrappers removed, and
record per step: correctness (strict metric), every confidence feature, input length, and whether the target element survives the
220-element cap. Feeds the pruning study and the learned deferral router (no frontier calls).
Usage (repo root): HF_HUB_OFFLINE=1 .venv/bin/python bench/ac_features.py"""
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "release/milo-pointer"), str(ROOT / "bench")]
from ac_frontier import gold_box, strict_ok  # noqa: E402
from milo_pointer import TYPES, MiloPointer, el_text  # noqa: E402
from parity_check import wanted_test_rows  # noqa: E402

MAXLEN, CAP = 3072, 220


def informative(e):
    """An element is kept by pruning if it shows text, a description or a hint, or can be acted on."""
    return bool(e["text"] or e["desc"] or e["hint"]) or any(e.get(k) for k in ("click", "long", "edit", "scroll", "checked"))


def centre_in(e, b):
    cx, cy = (e["b"][0] + e["b"][2]) / 2, (e["b"][1] + e["b"][3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def run(m, r, keep):
    """keep: original indices of the elements shown to the model, in order."""
    tok = m.tok
    prefix = f"{r['instr']} || {r['hist']} || goal: {r['goal']}"
    p = tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
    ids = [tok.cls_token_id] + p["input_ids"] + [tok.sep_token_id]
    offs = [(0, 0)] + p["offset_mapping"] + [(0, 0)]
    marks, shown = [], []
    for k in keep[:CAP]:
        t = tok(el_text(r["els"][k], r["w"], r["h"]), add_special_tokens=False)["input_ids"][:28]
        if len(ids) + len(t) + 2 > MAXLEN:
            break
        marks.append(len(ids))
        ids += [tok.mask_token_id] + t
        shown.append(k)
    ids.append(tok.sep_token_id)
    with torch.no_grad():
        lt, lp, lse = (x.float().cpu() for x in m.net(torch.tensor([ids], device=m.device)))
    pt = torch.softmax(lt, -1)
    top2 = pt.topk(2)
    ty = TYPES[int(top2.indices[0])]
    f = {"len": len(ids), "n_shown": len(shown), "type": ty, "p_type": float(top2.values[0]),
         "type_margin": float(top2.values[0] - top2.values[1]), "type_entropy": float(-(pt * (pt + 1e-12).log()).sum()),
         "p_el": 1.0, "el_margin": 1.0, "p_span": 1.0, "icon": 0}
    a = {"action_type": ty}
    if ty in ("click", "long_press") and marks:
        s = torch.softmax(lp[marks], -1)
        t2 = s.topk(min(2, len(marks)))
        j = int(t2.indices[0])
        f["p_el"] = float(t2.values[0])
        f["el_margin"] = float(t2.values[0] - (t2.values[1] if len(marks) > 1 else 0))
        e = r["els"][shown[j]]
        f["icon"] = int(not (e["text"] or e["desc"]))
        a["element"] = shown[j]
    elif ty in ("input_text", "open_app"):
        n = len(offs)
        ps, pe = torch.softmax(lse[:n, 0], -1), torch.softmax(lse[:n, 1], -1)
        s0 = int(ps.argmax())
        e0 = int(pe[s0:].argmax()) + s0
        f["p_span"] = float(ps[s0] * pe[e0])
        a["text" if ty == "input_text" else "app_name"] = prefix[offs[s0][0]:offs[e0][1]]
    f["conf"] = f["p_type"] * f["p_el"]
    f["ok"] = bool(strict_ok(r, a))
    gb = gold_box(r)
    f["gold_shown"] = None if not gb else any(centre_in(r["els"][k], gb) for k in shown)
    return f


if __name__ == "__main__":
    m = MiloPointer.load(ROOT / "release/milo-pointer")
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ac_sample1000", "ac_fresh1000"]
    S = {s: json.load(open(ROOT / f"bench/{s}.json")) for s in names}
    T = wanted_test_rows(set().union(*S.values()))
    out = {}
    t0 = time.time()
    for name, idx in S.items():
        rows = []
        for n, i in enumerate(idx):
            r = T[i]
            full = list(range(len(r["els"])))
            pruned = [k for k in full if informative(r["els"][k])]
            rows.append({"i": i, "gt": r["action"]["action_type"], "n_els": len(full), "n_pruned": len(pruned),
                         "full": run(m, r, full), "pruned": run(m, r, pruned)})
            if n % 250 == 0:
                print(name, n, f"{time.time() - t0:.0f}s", flush=True)
        out[name] = rows
    suffix = "" if len(sys.argv) == 1 else "_" + "_".join(names)
    json.dump(out, open(ROOT / f"bench/results/ac_features{suffix}.json", "w"))
    for name, rows in out.items():
        for v in ("full", "pruned"):
            ok = sum(r[v]["ok"] for r in rows)
            L = sorted(r[v]["len"] for r in rows)
            miss = sum(r[v]["gold_shown"] is False for r in rows)
            print(f"{name} {v:6}: correct {ok}  median tokens {L[len(L)//2]}  p90 {L[int(.9*len(L))]}  target cut off {miss}")
