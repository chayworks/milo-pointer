"""Export the released pointer to ONNX (fp32 and int8 dynamic) and check it against PyTorch on real AndroidControl inputs.
Usage (repo root): python release/milo-pointer/android/export_onnx.py"""

import json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parents[1]
sys.path[:0] = [str(PKG), str(ROOT / "bench"), str(PKG)]
from milo_pointer import MiloPointer, _Net
from parity_check import wanted_test_rows
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

cfg = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
cfg._attn_implementation = "eager"
net = _Net(cfg)
net.load_state_dict(
    {k: v.float() for k, v in load_file(str(PKG / "model.safetensors")).items()}, strict=True
)
net.eval()


class Wrap(torch.nn.Module):
    def __init__(s, n):
        super().__init__()
        s.n = n

    def forward(s, ids):
        hs = s.n.enc(input_ids=ids).last_hidden_state[0]
        return s.n.head_type(hs[0]), s.n.pt(hs).squeeze(-1), s.n.se(hs)


w = Wrap(net).eval()
dummy = torch.randint(5, 1000, (1, 600))
fp = HERE / "pointer_fp32.onnx"
torch.onnx.export(
    w,
    (dummy,),
    str(fp),
    input_names=["ids"],
    output_names=["type_logits", "elem_logits", "span_logits"],
    dynamic_axes={"ids": {1: "seq"}, "elem_logits": {0: "seq"}, "span_logits": {0: "seq"}},
    opset_version=17,
    dynamo=False,
)
from onnxruntime.quantization import quantize_dynamic, QuantType

q8 = HERE / "pointer_int8.onnx"
quantize_dynamic(str(fp), str(q8), weight_type=QuantType.QInt8)
print("sizes MB", round(fp.stat().st_size / 1e6), round(q8.stat().st_size / 1e6))

# agreement on real inputs: does each ONNX model pick the same action type and element as PyTorch?
import onnxruntime as ort

m = MiloPointer.load(PKG, device="cpu")
tok = m.tok
idx = json.load(open(ROOT / "bench/ac_sample1000.json"))[:40]
T = wanted_test_rows(set(idx))
from milo_pointer import el_text


def ids_of(r):
    ids = (
        [tok.cls_token_id]
        + tok(f"{r['instr']} || {r['hist']} || goal: {r['goal']}", add_special_tokens=False)["input_ids"]
        + [tok.sep_token_id]
    )
    marks = []
    for e in r["els"][:220]:
        t = tok(el_text(e, r["w"], r["h"]), add_special_tokens=False)["input_ids"][:28]
        if len(ids) + len(t) + 2 > 3072:
            break
        marks.append(len(ids))
        ids += [tok.mask_token_id] + t
    return ids + [tok.sep_token_id], marks


S = {
    k: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    for k, p in (("fp32", fp), ("int8", q8))
}
agree = {k: 0 for k in S}
saved = []
with torch.no_grad():
    for i in idx:
        ids, marks = ids_of(T[i])
        x = torch.tensor([ids])
        lt, lp, _ = (t.float() for t in m.net(x))
        ref = (int(lt.argmax()), int(lp[marks].argmax()) if marks else -1)
        for k, s in S.items():
            a, b, _ = s.run(None, {"ids": np.array([ids], dtype=np.int64)})
            agree[k] += (int(a.argmax()), int(b[marks].argmax()) if marks else -1) == ref
        saved.append({"i": i, "ids": ids, "marks": marks})
print("agreement with PyTorch (type+element) on 40 real steps:", agree)
json.dump(saved, open(HERE / "bench_inputs.json", "w"))
