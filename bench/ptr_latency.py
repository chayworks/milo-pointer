# On-device latency of the frozen 149M pointer (ModernBERT-base + heads) on the Mac M2, real AndroidControl test inputs.
import json, sys, time, torch, torch.nn as nn
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "bench"))
from ac_frontier import test_rows, el_text
from transformers import AutoTokenizer, AutoModel
MODEL = "answerdotai/ModernBERT-base"; tok = AutoTokenizer.from_pretrained(MODEL)
MARK = tok.convert_tokens_to_ids("[unused0]") if "[unused0]" in tok.get_vocab() else tok.mask_token_id
class Pointer(nn.Module):
    def __init__(self):
        super().__init__(); self.enc = AutoModel.from_pretrained(MODEL); h = self.enc.config.hidden_size
        self.head_type = nn.Linear(h, 11); self.pt = nn.Linear(h, 1); self.se = nn.Linear(h, 2)
    def forward(self, ids):
        hs = self.enc(input_ids=ids).last_hidden_state[0]; return self.head_type(hs[0]), self.pt(hs).squeeze(-1), self.se(hs)
m = Pointer(); sd = torch.load(ROOT / "training/ac/pointer_v2/pointer.pt", map_location="cpu")
print(m.load_state_dict(sd, strict=False)); dev = sys.argv[1]; m = m.to(dev).eval()
if dev == "mps": m = m.half()
T = test_rows(); idx = json.load(open(ROOT / "bench/ac_sample1000.json"))[:60]
def enc(r):
    ids = [tok.cls_token_id] + tok(f"{r['instr']} || {r['hist']} || goal: {r['goal']}", add_special_tokens=False)["input_ids"] + [tok.sep_token_id]
    for e in r["els"][:220]:
        t = tok(el_text(e, r["w"], r["h"]), add_special_tokens=False)["input_ids"][:28]
        if len(ids) + len(t) + 2 > 3072: break
        ids += [MARK] + t
    return ids + [tok.sep_token_id]
X = [enc(T[i]) for i in idx]; ts = []; ntok = []
with torch.no_grad():
    for k, ids in enumerate(X):
        x = torch.tensor([ids], device=dev); t0 = time.time(); out = m(x); _ = out[0].float().cpu()
        if k >= 5: ts.append(time.time() - t0); ntok.append(len(ids))
ts.sort(); ntok.sort()
print(dev, "median ms", round(1000 * ts[len(ts)//2]), "p90 ms", round(1000 * ts[int(.9*len(ts))]), "median tokens", ntok[len(ntok)//2])
