"""Milo Pointer: a 149M on-device model that picks the next UI action from an accessibility tree.

    from milo_pointer import MiloPointer
    m = MiloPointer.load("milo-pointer")            # folder with model.safetensors
    a = m.predict(goal="Turn on dark mode", instruction="Open display settings",
                  elements=[{"cls": "TextView", "text": "Display", "b": [0, 400, 1080, 520], "click": 1}, ...],
                  width=1080, height=2400)
    # {'action_type': 'click', 'element': 3, 'confidence': 0.998, 'defer': False}

Elements are dicts with: cls, text, desc, hint, rid (resource id), b = [left, top, right, bottom] in pixels, and 0/1 flags
click, long, edit, scroll, checked, sel. Only the first 220 elements are read. When `defer` is True the model is unsure
(confidence below the threshold registered for the AndroidControl router); send the step to a larger model instead.
"""

import re
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BASE = "answerdotai/ModernBERT-base"
TYPES = [
    "click",
    "long_press",
    "input_text",
    "scroll_up",
    "scroll_down",
    "scroll_left",
    "scroll_right",
    "navigate_back",
    "navigate_home",
    "open_app",
    "wait",
]
DEFER_BELOW = 0.9854
MAXLEN = 3072


def _words(rid):
    return " ".join(re.sub(r"([a-z])([A-Z])", r"\1 \2", rid).replace("_", " ").split()).lower()


def el_text(e, W, H):  # identical to the training rendering
    g = lambda k: e.get(k) or ""
    main = (g("text") or g("desc") or g("hint"))[:80]
    extra = [_words(g("rid"))] if g("rid") else []
    if g("desc") and g("text"):
        extra.append(g("desc")[:40])
    if g("hint") and (g("text") or g("desc")):
        extra.append("hint " + g("hint")[:30])
    x = (e["b"][0] + e["b"][2]) / 2 / max(W, 1)
    y = (e["b"][1] + e["b"][3]) / 2 / max(H, 1)
    flags = [
        f
        for f, k in (
            ("tap", "click"),
            ("edit", "edit"),
            ("scroll", "scroll"),
            ("checked", "checked"),
            ("selected", "sel"),
        )
        if e.get(k)
    ]
    return f"{g('cls')} {main} | {' | '.join(extra)} @{int(y*10)},{int(x*10)} {' '.join(flags)}"


class _Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc = AutoModel.from_config(cfg)
        h = cfg.hidden_size
        self.head_type = nn.Linear(h, len(TYPES))
        self.pt = nn.Linear(h, 1)
        self.se = nn.Linear(h, 2)

    def forward(self, ids):
        hs = self.enc(input_ids=ids).last_hidden_state[0]
        return self.head_type(hs[0]), self.pt(hs).squeeze(-1), self.se(hs)


class MiloPointer:
    def __init__(self, net, tok, device):
        self.net, self.tok, self.device = net, tok, device

    @classmethod
    def load(cls, path, device=None):
        from safetensors.torch import load_file
        from transformers import AutoConfig

        device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        p = Path(path)
        tok = AutoTokenizer.from_pretrained(p if (p / "tokenizer.json").exists() else BASE)
        cfg = AutoConfig.from_pretrained(p if (p / "config.json").exists() else BASE)
        net = _Net(cfg)
        net.load_state_dict(
            {k: v.float() for k, v in load_file(str(p / "model.safetensors")).items()}, strict=True
        )
        net = net.to(device).eval()
        if device != "cpu":
            net = net.half()
        return cls(net, tok, device)

    @torch.no_grad()
    def predict(self, goal, instruction, elements, width, height, last_step="new step"):
        tok = self.tok
        prefix = f"{instruction} || {last_step} || goal: {goal}"
        p = tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
        ids = [tok.cls_token_id] + p["input_ids"] + [tok.sep_token_id]
        offs = [(0, 0)] + p["offset_mapping"] + [(0, 0)]
        marks, kept = [], []
        for k, e in enumerate(elements[:220]):
            t = tok(el_text(e, width, height), add_special_tokens=False)["input_ids"][:28]
            if len(ids) + len(t) + 2 > MAXLEN:
                break
            marks.append(len(ids))
            ids += [tok.mask_token_id] + t
            kept.append(k)
        ids.append(tok.sep_token_id)
        lt, lp, lse = (x.float() for x in self.net(torch.tensor([ids], device=self.device)))
        pt = torch.softmax(lt, -1)
        conf = float(pt.max())
        ty = TYPES[int(pt.argmax())]
        out = {"action_type": ty}
        if ty in ("click", "long_press") and marks:
            sm = torch.softmax(lp[marks], -1)
            j = int(sm.argmax())
            conf *= float(sm[j])
            k = kept[j]
            eb = elements[k]["b"]
            out.update(element=k, x=(eb[0] + eb[2]) // 2, y=(eb[1] + eb[3]) // 2)
        elif ty in ("input_text", "open_app"):
            n = len(offs)
            s = int(lse[:n, 0].argmax())
            e = int(lse[s:n, 1].argmax()) + s
            out["text" if ty == "input_text" else "app_name"] = prefix[offs[s][0] : offs[e][1]]
        out["confidence"] = round(conf, 4)
        out["defer"] = conf < DEFER_BELOW
        return out
