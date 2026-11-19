"""AC-Planner2 (paper/PREREGISTRATION.md): a stronger planner (Qwen3-4B, QLoRA, longer history) in front of the clean 149M executor,
with the AC-PlanConf check refitted on top. Two Kaggle sessions: STAGE = "train" saves planner2_lora/; STAGE = "eval" (with the
train kernel's output attached) writes planner2_results.json. STAGE = "both" with SMOKE = True checks the whole pipeline on a few
hundred training steps and 100 dev steps (50 stand in for test); its numbers are not reported. No frontier model is used."""
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
STAGE = "train"
BASE_LM = "Qwen/Qwen3-4B-Instruct-2507"
N_TRAIN = 400 if SMOKE else 24000
SEED, K, TEMP = 20261118, 8, 0.8
N_PREV, N_ELS, MAXLEN = 8, 60, 768
TRAIN_BUDGET_H = 0.15 if SMOKE else 10.5
W = "/kaggle/working"

subprocess.run("pip uninstall -y -q torchao; pip install -q -U 'transformers>=4.51' peft accelerate bitsandbytes huggingface_hub",
               shell=True, check=True)  # Kaggle's torchao breaks peft
import numpy as np  # noqa: E402
import torch  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

t0 = time.time()
def log(*a):
    print(f"[{(time.time() - t0) / 60:6.1f} min]", *a, flush=True)

# ---- data -------------------------------------------------------------------------------------------------------------
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
for r in rows:  # the episode's previous recorded actions (offline, as in every AndroidControl evaluation)
    if r["ep"] != prev_ep:
        prev_ep, acts = r["ep"], []
    r["prev"] = " ; ".join(acts[-N_PREV:]) or "none"
    r["step_no"] = len(acts) + 1
    acts.append(act_str(r))


def screen(r, limit=N_ELS):
    seen, out = set(), []
    for e in r["els"]:
        t = (e["text"] or e["desc"] or e["hint"]).strip()
        if not t or t in seen or e.get("pkg", "").startswith("com.android.systemui"):
            continue
        seen.add(t)
        kind = "button" if e.get("click") else "field" if e.get("edit") else "scrollable" if e.get("scroll") else "text"
        out.append(f"[{kind}] {t[:60]}")
        if len(out) >= limit:
            break
    return "\n".join(out)


def prompt(r):
    return (f"Goal: {r['goal']}\nStep {r['step_no']}. Previous actions: {r['prev']}\nScreen:\n{screen(r)}\n"
            "Write the next step instruction for the phone, in the style of the dataset.\nInstruction:")


tok = AutoTokenizer.from_pretrained(BASE_LM)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True)

# ---- stage 1: QLoRA training ------------------------------------------------------------------------------------------
if STAGE in ("train", "both"):
    train = [r for r in rows if r["ep"] % 20 != 0]
    random.Random(SEED).shuffle(train)
    train = train[:N_TRAIN]
    log("planner training steps", len(train))
    lm = AutoModelForCausalLM.from_pretrained(BASE_LM, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=True)
    lm = get_peft_model(lm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in lm.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    def example(r):
        p = tok(prompt(r), add_special_tokens=False)["input_ids"]
        y = tok(" " + r["instr"].strip(), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        p = p[-(MAXLEN - len(y)):] if len(p) + len(y) > MAXLEN else p
        return p + y, [-100] * len(p) + y

    ex = sorted((example(r) for r in train), key=lambda x: len(x[0]))
    BS, ACC = 4, 8
    batches = [ex[i:i + BS] for i in range(0, len(ex), BS)]
    random.Random(SEED).shuffle(batches)
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    steps = math.ceil(len(batches) / ACC)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler()
    lm.train()
    for i, b in enumerate(batches):
        L = max(len(x) for x, _ in b)
        ids = torch.tensor([[tok.pad_token_id] * (L - len(x)) + x for x, _ in b]).cuda()
        lab = torch.tensor([[-100] * (L - len(y)) + y for _, y in b]).cuda()
        att = (torch.arange(L)[None, :] >= torch.tensor([L - len(x) for x, _ in b])[:, None]).long().cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lm(input_ids=ids, attention_mask=att, labels=lab).loss / ACC
        scaler.scale(loss).backward()
        if (i + 1) % ACC == 0 or i + 1 == len(batches):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in lm.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
        if i % 200 == 0:
            log(f"batch {i}/{len(batches)} loss {loss.item() * ACC:.3f}")
        if (time.time() - t0) / 3600 > TRAIN_BUDGET_H:
            log(f"time budget reached after batch {i + 1}/{len(batches)} ({(i + 1) * BS} examples)")
            break
    lm.save_pretrained(f"{W}/planner2_lora")
    json.dump({"examples_seen": (i + 1) * BS, "batches": i + 1, "of": len(batches)}, open(f"{W}/planner2_train.json", "w"))
    log("planner trained")
    del lm, opt
    torch.cuda.empty_cache()

if STAGE == "train":
    sys.exit(0)

# ---- stage 2: plans, executor, plan check -----------------------------------------------------------------------------
from huggingface_hub import snapshot_download  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

dev = [r for r in rows if r["ep"] % 40 == 0]
test = [r for r in rows if r["ep"] % 40 == 20]
if SMOKE:
    dev, test = dev[:50], dev[50:100]
log("dev steps", len(dev), "| test steps", len(test))

# ---- planner: greedy plan with token log-probabilities, then K samples -------------------------------------------------
adapter = os.path.dirname(sorted(glob.glob("/kaggle/input/**/planner2_lora/adapter_config.json", recursive=True) +
                                  glob.glob(f"{W}/planner2_lora/adapter_config.json"))[-1])
lm = AutoModelForCausalLM.from_pretrained(BASE_LM, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
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

prior = glob.glob("/kaggle/input/**/planconf_results.json", recursive=True)
c_old = [s["ok"] for s in json.load(open(prior[0]))["steps"]["test"]] if prior and not SMOKE else [r["ok"] for r in test]
assert len(c_old) == len(test)
y_new = [r["ok"] for r in test]
x = sum(a and not b for a, b in zip(y_new, c_old))
yv = sum(b and not a for a, b in zip(y_new, c_old))
n = len(test)
dd = (x - yv) / n
se = math.sqrt(max((x + yv) / n - dd * dd, 0) / n)
pa = min(1.0, 2 * sum(math.comb(x + yv, k) for k in range(min(x, yv) + 1)) / 2 ** (x + yv)) if x + yv else 1.0
h2a = {"new": sum(y_new) / n, "old": sum(c_old) / n, "diff": dd, "lo": dd - 1.96 * se, "hi": dd + 1.96 * se, "gained": x, "lost": yv, "p": pa}
log(f"H-P2a planner2 {h2a['new']:.3f} vs AC-Planner {h2a['old']:.3f}: {dd:+.3f} [{h2a['lo']:+.3f}, {h2a['hi']:+.3f}] "
    f"gained {x} lost {yv} p {pa:.2e} -> {'PASS' if h2a['lo'] > 0 and pa < 0.025 else 'see Holm'}")
norm2 = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731
log("exact match with human instruction", round(sum(norm2(r["plan"]) == norm2(r["instr"]) for r in test) / n, 3))

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
log(f"(secondary) agree vs exec AUROC agree {auroc(S['agree'], yt):.3f} vs exec {auroc(S['exec'], yt):.3f}: diff {d0:+.3f} [{lo1:+.3f}, {hi1:+.3f}]"
    f" boot p {p1:.4f} -> {'PASS' if lo1 > 0 else 'FAIL'}")
log(f"H-P2b comb >= {thr:.4f}: keeps {kept.mean():.3f} of test, accuracy {acc2:.3f} [{lo2:.3f}, {hi2:.3f}] -> "
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
           "H-P2a": h2a, "agree_vs_exec": {"diff": d0, "lo": lo1, "hi": hi1, "p": p1}, "H-P2b": {"thr": thr, "kept": float(kept.mean()), "acc": acc2,
           "lo": lo2, "hi": hi2}, "secondary": sec, "by_type": by_type,
           "steps": {name: [{"ep": r["ep"], "gt": r["action"]["action_type"], "instr": r["instr"], "plan": r["plan"],
                             "samples": r["samples"], **{f: r[f] for f in F + ["comb", "ok"]}} for r in st]
                     for name, st in (("dev", dev), ("test", test))}},
          open(f"{W}/planner2_results.json", "w"))
log("done")
