"""Re-score a Kaggle-trained Milo Pointer locally with BOTH protocols:
  strict (ours): click centre inside the gold element; typed text exact; open_app name fuzzy.
  OS-Atlas / UI-TARS protocol: click within 14% of screen width of the gold point; text token-F1 > 0.5; history given.
Reports on all 8,444 test steps and on the 7,868-step subset without wait/long_press (the set UI-R1 reports).
Uses the exact training-time source (encode/Pointer) pulled from the Kaggle kernel.
Usage: .venv/bin/python bench/pointer_eval_local.py <kernel_source.py> <pointer.pt> androidcontrol_text.jsonl.gz out.json"""
import gzip, json, math, re, sys, time
import torch

src_path, weights, data, out = sys.argv[1:5]
src = open(src_path).read()
pre = src.split("dev = \"cuda\"")[0]
pre = pre.replace('subprocess.run(', '(lambda *a, **k: 0)(')
pre = re.sub(r'path = glob\.glob\(.*?\)\[0\]', 'path = DATA', pre)
pre = pre.replace('rows = [json.loads(l) for l in gzip.open(path, "rt")]', 'rows = [json.loads(l) for l in gzip.open(path, "rt") if \'"split": "test"\' in l]')
g = {"DATA": data, "__name__": "x"}
exec(pre, g)
rows = g["rows"]; encode = g["encode"]; TYPES = g["TYPES"]; centre_in = g["centre_in"]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = g["Pointer"]().to(dev); model.load_state_dict(torch.load(weights, map_location="cpu")); model.eval()
if "--resume" not in sys.argv: pass


def f1(a, b):
    a, b = re.findall(r"\w+", (a or "").lower()), re.findall(r"\w+", (b or "").lower())
    if not a or not b: return float(a == b)
    c = sum(min(a.count(w), b.count(w)) for w in set(a)); p, r = c / len(a), c / len(b)
    return 2 * p * r / (p + r) if c else 0.0


def norm(s): return re.sub(r"\s+", " ", (s or "").lower()).strip(" .\"'")


res = []; t0 = time.time()
with torch.no_grad():
    for n, r in enumerate(rows):
        x = encode(r)
        lt, lp, lse = model(torch.tensor([x["ids"]], device=dev))
        ty = TYPES[int(lt.argmax())]; gt = TYPES[x["type"]]; a = r["action"]
        strict = lenient = ty == gt; pb = None; kids = []
        if ty == gt and gt in ("click", "long_press"):
            k = int(lp[x["marks"]].argmax()) if x["marks"] else -1
            e = x["kept"][k] if k >= 0 else None
            strict = bool(x["gb"]) and e is not None and centre_in(e, x["gb"])
            if e is not None:
                pb = e["b"]   # keep the predicted box + text children inside it, for tap-point analysis
                kids = [[c["b"], (c["text"] or c["desc"])[:40]] for c in x["kept"] if (c["text"] or c["desc"]) and c["b"][0] >= pb[0] and c["b"][1] >= pb[1] and c["b"][2] <= pb[2] and c["b"][3] <= pb[3]][:8]
                cx, cy = (e["b"][0] + e["b"][2]) / 2, (e["b"][1] + e["b"][3]) / 2
                lenient = math.hypot(cx - a["x"], cy - a["y"]) <= 0.14 * r["w"]
            else: lenient = False
        elif ty == gt and gt in ("input_text", "open_app"):
            nn_ = len(x["offs"]); s = int(lse[:nn_, 0].argmax()); e_ = int(lse[s:nn_, 1].argmax()) + s
            pred = x["prefix"][x["offs"][s][0]:x["offs"][e_][1]]
            want = a.get("text") if gt == "input_text" else a.get("app_name")
            strict = norm(pred) == norm(want) or (gt == "open_app" and len(norm(pred)) >= 3 and (norm(pred) in norm(want) or norm(want) in norm(pred)))
            lenient = f1(pred, want) > 0.5 or strict
        res.append({"ep": r["ep"], "gt": gt, "strict": bool(strict), "lenient": bool(lenient), "pb": pb, "kids": kids,
                    "gp": [a.get("x"), a.get("y")], "w": r["w"], "instr": r["instr"]})
        if n % 500 == 0: print(n, f"{time.time() - t0:.0f}s", flush=True)
json.dump(res, open(out, "w"))
def acc(rs, k): return f"{100 * sum(x[k] for x in rs) / max(len(rs), 1):.1f}% (n={len(rs)})"
sub = [x for x in res if x["gt"] not in ("wait", "long_press")]
print("ALL steps     strict", acc(res, "strict"), "| OS-Atlas protocol", acc(res, "lenient"))
print("no wait/long  strict", acc(sub, "strict"), "| OS-Atlas protocol", acc(sub, "lenient"))
