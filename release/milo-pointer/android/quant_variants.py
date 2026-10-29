"""Build int8 variants of the released pointer for the phone. The shipped model (pointer_int8.onnx) quantizes every supported op
per tensor; these variants keep more precision where 8 bits hurt most. Choose on ac_sample1000 with int8_router_eval.py.
Usage (repo root): .venv/bin/python release/milo-pointer/android/quant_variants.py"""

from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from safetensors.torch import load_file
from transformers import AutoConfig

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from milo_pointer import _Net  # noqa: E402

fp = HERE / "pointer_fp32.onnx"
if not fp.exists():
    cfg = AutoConfig.from_pretrained(HERE.parent)
    cfg._attn_implementation = "eager"
    net = _Net(cfg)
    net.load_state_dict({k: v.float() for k, v in load_file(str(HERE.parent / "model.safetensors")).items()}, strict=False)
    net.eval()

    class Wrap(torch.nn.Module):
        def __init__(s, n):
            super().__init__()
            s.n = n

        def forward(s, ids):
            hs = s.n.enc(input_ids=ids).last_hidden_state[0]
            return s.n.head_type(hs[0]), s.n.pt(hs).squeeze(-1), s.n.se(hs)

    torch.onnx.export(Wrap(net).eval(), (torch.randint(5, 1000, (1, 600)),), str(fp), input_names=["ids"],
                      output_names=["type_logits", "elem_logits", "span_logits"],
                      dynamic_axes={"ids": {1: "seq"}, "elem_logits": {0: "seq"}, "span_logits": {0: "seq"}},
                      opset_version=17, dynamo=False)
    del net

matmuls = [n.name for n in onnx.load(str(fp), load_external_data=False).graph.node if n.op_type in ("MatMul", "Gemm")]
last = [n for n in matmuls if "layers.21" in n or "head" in n or n.endswith("MatMul") and "enc" not in n]
VARIANTS = {
    "pc": dict(per_channel=True),  # per output channel scales
    "pc_mm": dict(per_channel=True, op_types_to_quantize=["MatMul"]),  # embeddings (Gather) stay fp32
    "pc_mm_last": dict(per_channel=True, op_types_to_quantize=["MatMul"], nodes_to_exclude=last),  # also last layer and heads
}
for name, kw in VARIANTS.items():
    out = HERE / f"pointer_int8_{name}.onnx"
    if not out.exists():
        quantize_dynamic(str(fp), str(out), weight_type=QuantType.QInt8, **kw)
    print(name, round(out.stat().st_size / 1e6), "MB", flush=True)
print("excluded in pc_mm_last:", len(last), "of", len(matmuls), "MatMul/Gemm nodes")
