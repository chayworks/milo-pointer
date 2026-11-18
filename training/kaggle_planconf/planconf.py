"""AC-PlanConf (paper/PREREGISTRATION.md): can cheap signals from the AC-Planner planner tell when its instruction is wrong?
Reuses the AC-Planner LoRA adapter (Kaggle output of chaitu6/milo-planner-distill) and the clean 149M executor. No frontier model.
Outputs (/kaggle/working): planconf_results.json (per-step signals and outcomes, test statistics), log lines on stdout.
Set SMOKE = True for a pipeline check on 100 dev steps (50 stand in for test); its numbers are not reported."""
import glob
import gzip
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter

SMOKE = False
BASE_LM = "Qwen/Qwen2.5-1.5B-Instruct"
K, TEMP, SEED = 8, 0.8, 20261118
MAXLEN = 512
W = "/kaggle/working"

subprocess.run("pip uninstall -y -q torchao; pip install -q -U 'transformers>=4.48' peft accelerate huggingface_hub", shell=True, check=True)  # Kaggle's torchao breaks peft
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from peft import PeftModel  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

t0 = time.time()
def log(*a):
    print(f"[{(time.time() - t0) / 60:6.1f} min]", *a, flush=True)

# ---- data (same construction as AC-Planner) ---------------------------------------------------------------------------
path = sorted(glob.glob("/kaggle/input/**/androidcontrol_text.jsonl*", recursive=True))[0]
rows = [json.loads(line) for line in (gzip.open(path, "rt") if path.endswith(".gz") else open(path))]
rows = [r for r in rows if r["split"] == "train"]


def act_str(r):
    a, t = r["action"], r["action"]["action_type"]
    if t in ("click", "long_press"):
        ins = [e for e in r["els"] if e["b"][0] <= a["x"] <= e["b"][2] and e["b"][1] <= a["y"] <= e["b"][3]]
        g = min(ins, key=lambda e: (e["b"][2] - e["b"][0]) * (e["b"][3] - e["b"][1])) if ins else None
        t += " " + ((g["text"] or g["desc"] or g["rid"])[:30] if g else "")
    elif t == "scroll":
        t += " " + a["direction"]
    elif t == "input_text":
        t += " " + a.get("text", "")[:30]
    elif t == "open_app":
        t += " " + a.get("app_name", "")
    return t.strip()


prev_ep, acts = None, []
for r in rows:
    if r["ep"] != prev_ep:
        prev_ep, acts = r["ep"], []
    r["prev"] = " ; ".join(acts[-4:]) or "none"
    acts.append(act_str(r))


def screen(r, limit=40):
    seen, out = set(), []
    for e in r["els"]:
        t = (e["text"] or e["desc"] or e["hint"]).strip()
        if not t or t in seen or e.get("pkg", "").startswith("com.android.systemui"):
            continue
        seen.add(t)
        kind = "button" if e.get("click") else "field" if e.get("edit") else "text"
        out.append(f"[{kind}] {t[:60]}")
        if len(out) >= limit:
            break
    return "\n".join(out)


def prompt(r):
    return (f"Goal: {r['goal']}\nPrevious actions: {r['prev']}\nScreen:\n{screen(r)}\n"
            "Write the next step instruction for the phone, in the style of the dataset.\nInstruction:")


dev = [r for r in rows if r["ep"] % 40 == 0]
test = [r for r in rows if r["ep"] % 40 == 20]
if SMOKE:
    dev, test = dev[:50], dev[50:100]
log("dev steps", len(dev), "| test steps", len(test))

# ---- planner: greedy plan with token log-probabilities, then K samples -------------------------------------------------
tok = AutoTokenizer.from_pretrained(BASE_LM)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
adapter = os.path.dirname(sorted(glob.glob("/kaggle/input/**/planner_lora/adapter_config.json", recursive=True))[0])
lm = AutoModelForCausalLM.from_pretrained(BASE_LM, torch_dtype=torch.float16).cuda()
lm = PeftModel.from_pretrained(lm, adapter).eval()
log("adapter", adapter)


def first_line(ids, lps=None):
    """Tokens of the first line (what the executor receives), stopping at end of text or a newline."""
    keep = []
    for j, t in enumerate(ids):
        t = int(t)
        if t in (tok.eos_token_id, tok.pad_token_id) or "\n" in tok.decode([t]):
            break
        keep.append(j)
    text = tok.decode([int(ids[j]) for j in keep], skip_special_tokens=True).strip()
    return text, ([float(lps[j]) for j in keep] if lps is not None else None)


def plan_all(steps):
    order = sorted(range(len(steps)), key=lambda i: len(prompt(steps[i])))
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        enc = tok([prompt(steps[i]) for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN - 40,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = lm.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id,
                              output_scores=True, return_dict_in_generate=True)
        sc = lm.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float().cpu().numpy()
        gen = out.sequences[:, enc["input_ids"].shape[1]:].cpu().numpy()
        for b, i in enumerate(idx):
            text, lps = first_line(gen[b], sc[b])
            steps[i]["plan"] = text
            steps[i]["lp_mean"] = float(np.mean(lps)) if lps else -20.0
            steps[i]["lp_min"] = float(np.min(lps)) if lps else -20.0
    torch.manual_seed(SEED)
    for s in range(0, len(order), 8):
        idx = order[s:s + 8]
        enc = tok([prompt(steps[i]) for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN - 40,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = lm.generate(**enc, max_new_tokens=40, do_sample=True, temperature=TEMP, top_p=1.0, top_k=0,
                              num_return_sequences=K, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:].cpu().numpy()
        for b, i in enumerate(idx):
            steps[i]["samples"] = [first_line(gen[b * K + k])[0] for k in range(K)]


plan_all(dev)
log("dev plans done")
plan_all(test)
log("test plans done; example:", test[0]["goal"], "->", test[0]["plan"], "| samples:", test[0]["samples"][:3])
prior = glob.glob("/kaggle/input/**/planner_results.json", recursive=True)
if prior and not SMOKE:
    old = json.load(open(prior[0]))["steps"]
    same = sum(o["plan"].strip() == r["plan"].strip() for o, r in zip(old, test)) / len(test)
    log(f"determinism check: greedy test plans equal to AC-Planner's on {100 * same:.1f}% of steps")
del lm
torch.cuda.empty_cache()

# ---- executor ---------------------------------------------------------------------------------------------------------
R = f"{W}/repo"
subprocess.run(f"git clone -q --depth 1 https://github.com/chayworks/milo-pointer {R}", shell=True, check=True)
snapshot_download("coolnow/milo-pointer", local_dir=f"{R}/release/milo-pointer",
                  allow_patterns=["model.safetensors", "config.json", "tokenizer*", "special_tokens_map.json", "clean_r3/*"])
os.makedirs(f"{R}/training/ac/pointer_clean_r3", exist_ok=True)
os.replace(f"{R}/release/milo-pointer/clean_r3/pointer_clean_r3.pt", f"{R}/training/ac/pointer_clean_r3/pointer_clean_r3.pt")
sys.path[:0] = [f"{R}/release/milo-pointer", f"{R}/bench"]
import ac_features  # noqa: E402
from sl_prepare import keep_r3, load_clean  # noqa: E402

_last = {}
_strict = ac_features.strict_ok
def _capture(r, a):  # the executor's chosen action, recorded on its way to the scorer
    _last["a"] = dict(a)
    return _strict(r, a)
ac_features.strict_ok = _capture
m = load_clean()
norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()  # noqa: E731


def action_key(a):
    t = a["action_type"]
    if "element" in a:
        return (t, a["element"])
    if "text" in a or "app_name" in a:
        return (t, norm(a.get("text") or a.get("app_name")))
    return (t,)


def hist_strings(instrs, rs):
    """Same history/consequence string as training, computed from the greedy plans (AC-Planner arm C)."""
    def sig(r):
        return Counter((e["text"] or e["desc"] or e["rid"], e["cls"]) for e in r["els"])
    out, prev = [], None
    for ins, r in zip(instrs, rs):
        s = sig(r)
        if prev and prev[0] == r["ep"] and prev[1] == ins.strip():
            j = sum((s & prev[3]).values()) / max(sum((s | prev[3]).values()), 1)
            out.append(f"repeat of last step (last action {prev[2]}), screen {int(round(j * 10)) * 10}% unchanged")
        else:
            out.append("new step" + (f" (last action {prev[2]})" if prev and prev[0] == r["ep"] else ""))
        prev = (r["ep"], ins.strip(), r["action"]["action_type"], s)
    return out


def execute(steps):
    hs = hist_strings([r["plan"] for r in steps], steps)
    for r, h in zip(steps, hs):
        keep = [k for k, e in enumerate(r["els"]) if keep_r3(e)]
        f = ac_features.run(m, {**r, "instr": r["plan"], "hist": h}, keep)
        r["ok"], r["exec"], g = bool(f["ok"]), float(f["conf"]), action_key(_last["a"])
        same = 0
        for s in r["samples"]:
            ac_features.run(m, {**r, "instr": s, "hist": h}, keep)
            same += action_key(_last["a"]) == g
        r["agree"] = same / K


execute(dev)
log("dev executed")
execute(test)
log("test executed")

# ---- statistics -------------------------------------------------------------------------------------------------------
F = ["exec", "lp_mean", "lp_min", "agree"]
X = lambda st: np.array([[r[f] for f in F] for r in st])  # noqa: E731
yd, yt = np.array([r["ok"] for r in dev], int), np.array([r["ok"] for r in test], int)
sc = StandardScaler().fit(X(dev))
clf = LogisticRegression(max_iter=1000).fit(sc.transform(X(dev)), yd)
for st in (dev, test):
    for r, p in zip(st, clf.predict_proba(sc.transform(X(st)))[:, 1]):
        r["comb"] = float(p)
log("comb coefficients", dict(zip(F, np.round(clf.coef_[0], 3))))


def auroc(s, y):
    if y.min() == y.max():
        return float("nan")
    rk = rankdata(s)
    n1 = y.sum()
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def wilson(k, n):
    if n == 0:
        return (float("nan"),) * 3
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


S = {f: np.array([r[f] for r in test]) for f in F + ["comb"]}
Sd = {f: np.array([r[f] for r in dev]) for f in F + ["comb"]}
eps = np.array([r["ep"] for r in test])
uniq = np.unique(eps)
pos = {e: np.where(eps == e)[0] for e in uniq}
rng = np.random.default_rng(SEED)
diffs = []
for _ in range(2000):
    ii = np.concatenate([pos[e] for e in rng.choice(uniq, len(uniq))])
    diffs.append(auroc(S["agree"][ii], yt[ii]) - auroc(S["exec"][ii], yt[ii]))
diffs = np.array([d for d in diffs if not math.isnan(d)])
d0 = auroc(S["agree"], yt) - auroc(S["exec"], yt)
lo1, hi1 = np.percentile(diffs, [2.5, 97.5])
p1 = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
thr = float(np.quantile(Sd["comb"], 0.5))
kept = S["comb"] >= thr
acc2, lo2, hi2 = wilson(int(yt[kept].sum()), int(kept.sum()))
log(f"base accuracy test {yt.mean():.3f} dev {yd.mean():.3f}")
log(f"H-PC1 AUROC agree {auroc(S['agree'], yt):.3f} vs exec {auroc(S['exec'], yt):.3f}: diff {d0:+.3f} [{lo1:+.3f}, {hi1:+.3f}]"
    f" boot p {p1:.4f} -> {'PASS' if lo1 > 0 else 'FAIL'}")
log(f"H-PC2 comb >= {thr:.4f}: keeps {kept.mean():.3f} of test, accuracy {acc2:.3f} [{lo2:.3f}, {hi2:.3f}] -> "
    f"{'PASS' if lo2 >= 0.70 else 'FAIL'}")
sec = {"auroc": {f: auroc(S[f], yt) for f in S}, "coverage": {}}
for f in S:
    for c in (0.3, 0.5, 0.7):
        t = float(np.quantile(Sd[f], 1 - c))
        k = S[f] >= t
        sec["coverage"][f"{f}@{c}"] = {"thr": t, "test_coverage": float(k.mean()), "acc": wilson(int(yt[k].sum()), int(k.sum()))}
    log(f"{f:8} AUROC {sec['auroc'][f]:.3f} | " + " | ".join(
        f"{c:.0%}: cov {sec['coverage'][f'{f}@{c}']['test_coverage']:.2f} acc {sec['coverage'][f'{f}@{c}']['acc'][0]:.3f}"
        for c in (0.3, 0.5, 0.7)))
by_type = {}
for t in sorted({r["action"]["action_type"] for r in test}):
    ix = np.array([r["action"]["action_type"] == t for r in test])
    by_type[t] = {"n": int(ix.sum()), "acc": float(yt[ix].mean()), "kept": float(kept[ix].mean()),
                  "acc_kept": float(yt[ix & kept].mean()) if (ix & kept).any() else None}
log("per action", by_type)
json.dump({"smoke": SMOKE, "n_dev": len(dev), "n_test": len(test), "features": F, "coef": clf.coef_[0].tolist(),
           "H-PC1": {"diff": d0, "lo": lo1, "hi": hi1, "p": p1}, "H-PC2": {"thr": thr, "kept": float(kept.mean()), "acc": acc2,
           "lo": lo2, "hi": hi2}, "secondary": sec, "by_type": by_type,
           "steps": {name: [{"ep": r["ep"], "gt": r["action"]["action_type"], "instr": r["instr"], "plan": r["plan"],
                             "samples": r["samples"], **{f: r[f] for f in F + ["comb", "ok"]}} for r in st]
                     for name, st in (("dev", dev), ("test", test))}},
          open(f"{W}/planconf_results.json", "w"))
log("done")
