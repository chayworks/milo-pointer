
import glob, gzip, json, math, random, re, time, os, subprocess
subprocess.run("pip install -q -U 'transformers>=4.48'", shell=True)
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
MODEL = "answerdotai/ModernBERT-base"; MAXLEN = 3072; EPOCHS = 2; LR = 5e-5; ACC = 8
torch.manual_seed(1); random.seed(1)
path = glob.glob("/kaggle/input/**/androidcontrol_text.jsonl.gz", recursive=True)[0]
rows = [json.loads(l) for l in gzip.open(path, "rt")]
print(len(rows), "steps", {s: sum(r["split"] == s for r in rows) for s in ("train", "test")})
from collections import Counter
print(Counter(r["action"]["action_type"] for r in rows))
print([r["action"] for r in rows if r["action"]["action_type"] == "input_text"][:3])

from collections import Counter as _C
def _sig(r): return _C((e["text"] or e["desc"] or e["rid"], e["cls"]) for e in r["els"])
_prev = None
for r in rows:   # history + consequence (as the paper's models get history): same instruction as last step? did the screen change?
    s = _sig(r)
    if _prev and _prev[0] == r["ep"] and _prev[1] == r["instr"].strip():
        j = sum((s & _prev[3]).values()) / max(sum((s | _prev[3]).values()), 1)
        r["hist"] = f"repeat of last step (last action {_prev[2]}), screen {int(round(j * 10)) * 10}% unchanged"
    else:
        r["hist"] = "new step" + (f" (last action {_prev[2]})" if _prev and _prev[0] == r["ep"] else "")
    _prev = (r["ep"], r["instr"].strip(), r["action"]["action_type"], s)

TYPES = ["click", "long_press", "input_text", "scroll_up", "scroll_down", "scroll_left", "scroll_right", "navigate_back",
         "navigate_home", "open_app", "wait"]
def tlabel(a):
    t = a["action_type"]
    return TYPES.index("scroll_" + a["direction"]) if t == "scroll" else TYPES.index(t)

def words(rid):
    return " ".join(re.sub(r"([a-z])([A-Z])", r"\1 \2", rid).replace("_", " ").split()).lower()

def el_text(e, W, H):
    main = (e["text"] or e["desc"] or e["hint"])[:80]
    extra = [words(e["rid"])] if e["rid"] else []
    if e["desc"] and e["text"]: extra.append(e["desc"][:40])
    if e["hint"] and (e["text"] or e["desc"]): extra.append("hint " + e["hint"][:30])
    x = (e["b"][0] + e["b"][2]) / 2 / max(W, 1); y = (e["b"][1] + e["b"][3]) / 2 / max(H, 1)
    flags = [f for f, k in (("tap", "click"), ("edit", "edit"), ("scroll", "scroll"), ("checked", "checked"), ("selected", "sel")) if e.get(k)]
    return f"{e['cls']} {main} | {' | '.join(extra)} @{int(y*10)},{int(x*10)} {' '.join(flags)}"

def gold_elements(r):
    a = r["action"]
    if a["action_type"] not in ("click", "long_press"): return None
    px, py = a["x"], a["y"]
    inside = [e for e in r["els"] if e["b"][0] <= px <= e["b"][2] and e["b"][1] <= py <= e["b"][3]]
    if not inside: return None
    g = min(inside, key=lambda e: (e["b"][2] - e["b"][0]) * (e["b"][3] - e["b"][1]))
    return g["b"]

def centre_in(e, b):
    cx = (e["b"][0] + e["b"][2]) / 2; cy = (e["b"][1] + e["b"][3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]

tok = AutoTokenizer.from_pretrained(MODEL)
MARK = tok.mask_token_id

def span_target(prefix, target, enc_offsets):
    if not target: return None
    i = prefix.lower().find(target.lower().strip()); j = i + len(target.strip())
    if i < 0:   # "Zoho Meet" said, "Zoho Meeting" gold: longest run of said words that start target words
        tw = [w.lower() for w in re.findall(r"\w+", target)]; best = None
        ms = list(re.finditer(r"\w+", prefix.split(" || ")[0]))
        for a0 in range(len(ms)):
            b0 = a0
            while b0 < len(ms) and any(t.startswith(ms[b0].group().lower()) and len(ms[b0].group()) >= 2 for t in tw): b0 += 1
            if b0 > a0 and (best is None or b0 - a0 > best[1] - best[0]): best = (a0, b0)
        if not best or (best[1] - best[0]) < max(1, len(tw) // 2): return None
        i, j = ms[best[0]].start(), ms[best[1] - 1].end()
    s = next((k for k, (a, b) in enumerate(enc_offsets) if a <= i < b), None)
    e = next((k for k, (a, b) in enumerate(enc_offsets) if a < j <= b), None)
    return (s, e) if s is not None and e is not None else None

LAYOUT = {"FrameLayout", "LinearLayout", "RelativeLayout", "ViewGroup", "ConstraintLayout", "CoordinatorLayout", "View", "ScrollView",
          "HorizontalScrollView", "RecyclerView", "ListView", "ViewPager", "DrawerLayout", "LinearLayoutCompat", "CardView"}
def keep_r3(e):
    empty = not (e["text"] or e["desc"] or e["hint"]); inert = not any(e.get(k) for k in ("click", "long", "edit", "scroll", "checked"))
    return not (e["cls"] in LAYOUT and empty and inert)

def encode(r):
    prefix = f"{r['instr']} || {r['hist']} || goal: {r['goal']}"
    p = tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
    ids = [tok.cls_token_id] + p["input_ids"] + [tok.sep_token_id]
    offs = [(0, 0)] + p["offset_mapping"] + [(0, 0)]
    marks = []; kept = []
    for e in [x for x in r["els"] if keep_r3(x)][:220]:   # R3: drop empty, inert layout containers
        t = tok(el_text(e, r["w"], r["h"]), add_special_tokens=False)["input_ids"][:28]
        if len(ids) + len(t) + 2 > MAXLEN: break
        marks.append(len(ids)); ids += [MARK] + t; kept.append(e)
    ids.append(tok.sep_token_id)
    a = r["action"]; gb = gold_elements(r)
    pos = [k for k, e in enumerate(kept) if gb and centre_in(e, gb)]
    txt = a.get("text") if a["action_type"] == "input_text" else a.get("app_name") if a["action_type"] == "open_app" else None
    sp = span_target(prefix, txt, offs) if txt else None
    return {"ids": ids, "marks": marks, "kept": kept, "type": tlabel(a), "pos": pos, "span": sp, "offs": offs, "prefix": prefix, "gb": gb}

class Pointer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        h = self.enc.config.hidden_size
        self.head_type = nn.Linear(h, len(TYPES)); self.pt = nn.Linear(h, 1); self.se = nn.Linear(h, 2)
    def forward(self, ids):
        hs = self.enc(input_ids=ids).last_hidden_state[0]
        return self.head_type(hs[0]), self.pt(hs).squeeze(-1), self.se(hs)

dev = "cuda"
model = Pointer().to(dev)
# CLEAN protocol: 5% of TRAIN episodes held out as validation (never trained on); every decision (threshold) uses validation only
train = [r for r in rows if r["split"] == "train" and r["ep"] % 20 != 0]; val = [r for r in rows if r["split"] == "train" and r["ep"] % 20 == 0]
test = [r for r in rows if r["split"] == "test"]; print("train", len(train), "val", len(val), "test", len(test))
t0 = time.time(); TR = [encode(r) for r in train]; TE = [encode(r) for r in test]; VA = [encode(r) for r in val]
print("encoded", time.time() - t0, "s; test tokens median", sorted(len(x["ids"]) for x in TE)[len(TE)//2],
      "; click steps with gold element kept:", sum(1 for x in TE if x["type"] in (0, 1) and x["pos"]), "/", sum(1 for x in TE if x["type"] in (0, 1)),
      "; text spans found:", sum(1 for x in TE if x["type"] in (2, 9) and x["span"]), "/", sum(1 for x in TE if x["type"] in (2, 9)))

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total = EPOCHS * len(TR) // ACC
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 200) * max(0.05, 1 - s / max(total, 1)))
scaler = torch.cuda.amp.GradScaler()

def loss_of(x):
    ids = torch.tensor([x["ids"]], device=dev)
    with torch.autocast("cuda", dtype=torch.float16):
        lt, lp, lse = model(ids)
    lt, lp, lse = lt.float(), lp.float(), lse.float()
    L = F.cross_entropy(lt[None], torch.tensor([x["type"]], device=dev))
    if x["pos"] and x["type"] in (0, 1):
        m = lp[x["marks"]]; L = L + (torch.logsumexp(m, 0) - torch.logsumexp(m[x["pos"]], 0))
    if x["span"]:
        s, e = x["span"]; n = len(x["offs"])
        L = L + 0.5 * (F.cross_entropy(lse[:n, 0][None], torch.tensor([s], device=dev)) + F.cross_entropy(lse[:n, 1][None], torch.tensor([e], device=dev)))
    return L

step = 0
for ep in range(EPOCHS):
    random.shuffle(TR); model.train(); run = 0
    for i, x in enumerate(TR):
        if time.time() - t0 > 9.5 * 3600: break   # stay inside Kaggle's 12 h limit; evaluate what we have
        L = loss_of(x) / ACC
        scaler.scale(L).backward(); run += L.item()
        if (i + 1) % ACC == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step(); step += 1
            if step % 200 == 0: print(f"ep {ep} step {step}/{total} loss {run/200:.3f} {time.time()-t0:.0f}s", flush=True); run = 0

def norm(s): return re.sub(r"\s+", " ", (s or "").lower()).strip(" .\"'")
def run(X, R):   # strict metric exactly as bench/pointer_eval_local.py (no clickable-ancestor lift) + router confidence
    out = []
    with torch.no_grad():
        for x, r in zip(X, R):
            with torch.autocast("cuda", dtype=torch.float16):
                lt, lp, lse = model(torch.tensor([x["ids"]], device=dev))
            lt, lp, lse = lt.float(), lp.float(), lse.float()
            pt = torch.softmax(lt, -1); ty = TYPES[int(pt.argmax())]; gt = TYPES[x["type"]]; a = r["action"]; conf = float(pt.max())
            if ty in ("click", "long_press") and x["marks"]:
                sm = torch.softmax(lp[x["marks"]], -1); conf *= float(sm.max())
            ok = ty == gt
            if ok and gt in ("click", "long_press"):
                k = int(lp[x["marks"]].argmax()) if x["marks"] else -1
                ok = bool(x["gb"]) and k >= 0 and centre_in(x["kept"][k], x["gb"])
            elif ok and gt in ("input_text", "open_app"):
                n = len(x["offs"]); s = int(lse[:n, 0].argmax()); e = int(lse[s:n, 1].argmax()) + s
                pred = x["prefix"][x["offs"][s][0]:x["offs"][e][1]]; want = a.get("text") if gt == "input_text" else a.get("app_name")
                ok = norm(pred) == norm(want) or (gt == "open_app" and len(norm(pred)) >= 3 and (norm(pred) in norm(want) or norm(want) in norm(pred)))
            out.append({"ep": r["ep"], "sub": r.get("sub"), "gt": gt, "ok": bool(ok), "conf": conf})
    return out
model.eval(); rv = run(VA, val); rt = run(TE, test)
cs = sorted(x["conf"] for x in rv); thr = cs[int(0.3 * len(cs))]   # threshold = 30th percentile on VALIDATION
summary = {"val_acc": sum(x["ok"] for x in rv) / len(rv), "test_acc": sum(x["ok"] for x in rt) / len(rt), "thr_val30": thr,
           "test_local_share": sum(x["conf"] >= thr for x in rt) / len(rt), "minutes": (time.time() - t0) / 60}
print(json.dumps(summary, indent=1))
json.dump({"summary": summary, "val": rv, "test": rt}, open("/kaggle/working/clean_r3_results.json", "w"))
torch.save(model.state_dict(), "/kaggle/working/pointer_clean_r3.pt")
