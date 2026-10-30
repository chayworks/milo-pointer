"""Accuracy and routing of the phone-ready ONNX model (int8 or fp32) on the 1,000 fresh steps, scored exactly like the frozen PyTorch
outputs (strict metric, no clickable-ancestor lift) and routed with the registered threshold to the registered Opus/Sonnet run-1 outputs.
Usage (repo root): python release/milo-pointer/android/int8_router_eval.py pointer_int8.onnx [sample=ac_fresh1000]
(model variants are chosen on ac_sample1000, the calibration sample, and then reported once on ac_fresh1000)"""

import json, math, os, sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(ROOT / "bench")]  # the package's modules take precedence
from ac_frontier import strict_ok
from ac_router_score import THR, mcnemar, strict
from parity_check import wanted_test_rows
from milo_pointer import TYPES, el_text
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(HERE.parent)  # tokenizer files ship with the package
SAMPLE = sys.argv[2] if len(sys.argv) > 2 else "ac_fresh1000"
idx = json.load(open(ROOT / f"bench/{SAMPLE}.json"))
N = len(idx)
T = wanted_test_rows(set(idx))
so = ort.SessionOptions()
so.intra_op_num_threads = 4
S = ort.InferenceSession(str(HERE / sys.argv[1]), so, providers=["CPUExecutionProvider"])


def sm(x):
    e = np.exp(x - x.max())
    return e / e.sum()


ok, cf, ts = [], [], []
for n, i in enumerate(idx):
    r = T[i]
    prefix = f"{r['instr']} || {r['hist']} || goal: {r['goal']}"
    p = tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
    ids = [tok.cls_token_id] + p["input_ids"] + [tok.sep_token_id]
    offs = [(0, 0)] + p["offset_mapping"] + [(0, 0)]
    marks = []
    for e in r["els"][:220]:
        t = tok(el_text(e, r["w"], r["h"]), add_special_tokens=False)["input_ids"][:28]
        if len(ids) + len(t) + 2 > 3072:
            break
        marks.append(len(ids))
        ids += [tok.mask_token_id] + t
    ids.append(tok.sep_token_id)
    t0 = time.time()
    lt, lp, lse = S.run(None, {"ids": np.array([ids], dtype=np.int64)})
    ts.append(time.time() - t0)
    pt = sm(lt)
    ty = TYPES[int(pt.argmax())]
    c = float(pt.max())
    a = {"action_type": ty}
    if ty in ("click", "long_press") and marks:
        s = sm(lp[marks])
        k = int(s.argmax())
        c *= float(s[k])
        a["element"] = k  # kept elements are the first len(marks) elements
    elif ty in ("input_text", "open_app"):
        L = len(offs)
        s0 = int(lse[:L, 0].argmax())
        e0 = int(lse[s0:L, 1].argmax()) + s0
        a["text" if ty == "input_text" else "app_name"] = prefix[offs[s0][0] : offs[e0][1]]
    ok.append(bool(strict_ok(r, a)))
    cf.append(c)
    if n % 200 == 0:
        print(n, flush=True)
ref = [bool(strict[i]) for i in idx]
loc = [c >= THR for c in cf]
print(
    f"{sys.argv[1]}: accuracy {sum(ok)} vs PyTorch {sum(ref)}; same outcome {sum(a == b for a, b in zip(ok, ref))}/{N}; local share {100*sum(loc)/N:.1f}%; median CPU s {sorted(ts)[N // 2]:.2f} ({SAMPLE})"
)
for m in ("opus", "sonnet"):
    fp = ROOT / f"bench/results/acfs_claude-{m}-5-5_{SAMPLE}.json"
    if not fp.exists():
        continue
    F = [x["ok"] for x in json.load(open(fp))["rows"]]
    cas = [p if l else f for p, f, l in zip(ok, F, loc)]
    x = sum(c and not f for c, f in zip(cas, F))
    y = sum(f and not c for c, f in zip(cas, F))
    d = (x - y) / N
    se = math.sqrt((x + y) / N - d * d) / math.sqrt(N)
    print(
        f"  int8 -> {m}: {sum(cas)} vs {m} alone {sum(F)}: {100*d:+.1f} [{100*(d-1.96*se):+.1f}, {100*(d+1.96*se):+.1f}] p={mcnemar(x, y):.2g}"
    )
json.dump({"ok": ok, "conf": cf}, open(HERE / f"router_eval_{Path(sys.argv[1]).stem}_{SAMPLE}.json", "w"))
