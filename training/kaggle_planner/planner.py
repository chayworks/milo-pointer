"""AC-Planner (paper/PREREGISTRATION.md): train a small planner on AndroidControl's human step instructions, then let the clean
149M executor act on its instructions from the goal alone. Runs end to end on one Kaggle T4; no frontier model is used.
Outputs (/kaggle/working): planner_results.json (per-step outcomes for arms A, B, C), planner_lora/ (adapter), log lines on stdout.
Set SMOKE = True for a pipeline check on a few hundred training steps and 50 test steps."""
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
N_TRAIN = 400 if SMOKE else 40000
N_TEST = 50 if SMOKE else None
SEED = 20261030
MAXLEN = 512
W = "/kaggle/working"

subprocess.run("pip uninstall -y -q torchao; pip install -q -U 'transformers>=4.48' peft accelerate huggingface_hub", shell=True, check=True)  # Kaggle's torchao breaks peft
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

t0 = time.time()
def log(*a):
    print(f"[{(time.time() - t0) / 60:6.1f} min]", *a, flush=True)

# ---- data -----------------------------------------------------------------------------------------------------------
path = sorted(glob.glob("/kaggle/input/**/androidcontrol_text.jsonl*", recursive=True))[0]
rows = [json.loads(line) for line in (gzip.open(path, "rt") if path.endswith(".gz") else open(path))]
rows = [r for r in rows if r["split"] == "train"]
log(len(rows), "train-split steps")


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


train = [r for r in rows if r["ep"] % 20 != 0]
random.Random(SEED).shuffle(train)
train = train[:N_TRAIN]
test = [r for r in rows if r["ep"] % 40 == 20]
if N_TEST:
    test = test[:N_TEST]
log("planner training steps", len(train), "| test steps", len(test))

# ---- planner training -------------------------------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(BASE_LM)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
lm = AutoModelForCausalLM.from_pretrained(BASE_LM, torch_dtype=torch.float16).cuda()
lm.gradient_checkpointing_enable()
lm.enable_input_require_grads()
lm = get_peft_model(lm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
for p in lm.parameters():
    if p.requires_grad:
        p.data = p.data.float()  # LoRA weights in fp32, base in fp16 (T4 has no bf16)


def example(r):
    p = tok(prompt(r), add_special_tokens=False)["input_ids"]
    y = tok(" " + r["instr"].strip(), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    p = p[-(MAXLEN - len(y)):] if len(p) + len(y) > MAXLEN else p
    return p + y, [-100] * len(p) + y


ex = sorted((example(r) for r in train), key=lambda x: len(x[0]))
BS, ACC = 8, 4
batches = [ex[i:i + BS] for i in range(0, len(ex), BS)]
random.Random(SEED).shuffle(batches)
opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
steps = math.ceil(len(batches) / ACC)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
scaler = torch.cuda.amp.GradScaler()
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
lm.save_pretrained(f"{W}/planner_lora")
log("planner trained")

# ---- planner inference on the test steps -----------------------------------------------------------------------------
lm.eval()
lm.gradient_checkpointing_disable()
gen = []
order = sorted(range(len(test)), key=lambda i: len(prompt(test[i])))
for s in range(0, len(order), 32):
    idx = order[s:s + 32]
    enc = tok([prompt(test[i]) for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN - 40,
              add_special_tokens=False).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = lm.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
    for i, o in zip(idx, out[:, enc["input_ids"].shape[1]:]):
        test[i]["plan"] = tok.decode(o, skip_special_tokens=True).strip().split("\n")[0]
log("planner instructions written; example:", test[0]["goal"], "->", test[0]["plan"], "| human:", test[0]["instr"])
del lm
torch.cuda.empty_cache()

# ---- executor: the released clean retrain, from the public repo and model page --------------------------------------
R = f"{W}/repo"
subprocess.run(f"git clone -q --depth 1 https://github.com/chayworks/milo-pointer {R}", shell=True, check=True)
snapshot_download("coolnow/milo-pointer", local_dir=f"{R}/release/milo-pointer",
                  allow_patterns=["model.safetensors", "config.json", "tokenizer*", "special_tokens_map.json", "clean_r3/*"])
os.makedirs(f"{R}/training/ac/pointer_clean_r3", exist_ok=True)
os.replace(f"{R}/release/milo-pointer/clean_r3/pointer_clean_r3.pt", f"{R}/training/ac/pointer_clean_r3/pointer_clean_r3.pt")
sys.path[:0] = [f"{R}/release/milo-pointer", f"{R}/bench"]
from ac_features import run  # noqa: E402
from sl_prepare import keep_r3, load_clean  # noqa: E402

m = load_clean()
TAU = 0.9842


def hist_strings(instrs, rs):
    """Same history/consequence string as training, computed from the arm's own instruction sequence."""
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


arms = {"A_human": [r["instr"] for r in test], "B_goal": [r["goal"] for r in test], "C_planner": [r["plan"] for r in test]}
res = {}
for name, instrs in arms.items():
    hs = hist_strings(instrs, test) if name != "B_goal" else [
        "new step" + (f" (last action {test[k - 1]['action']['action_type']})" if k and test[k - 1]["ep"] == r["ep"] else "")
        for k, r in enumerate(test)]
    out = []
    for r, ins, h in zip(test, instrs, hs):
        keep = [k for k, e in enumerate(r["els"]) if keep_r3(e)]
        f = run(m, {**r, "instr": ins, "hist": h}, keep)
        out.append({"ok": bool(f["ok"]), "conf": float(f["conf"])})
    res[name] = out
    n, ok = len(out), sum(x["ok"] for x in out)
    loc = [x for x in out if x["conf"] >= TAU]
    log(f"{name}: {ok}/{n} = {100 * ok / n:.1f}% | local {100 * len(loc) / n:.1f}% at {100 * sum(x['ok'] for x in loc) / max(len(loc), 1):.1f}%")


def mcn(a, b):
    x = sum(p and not q for p, q in zip(a, b))
    y = sum(q and not p for p, q in zip(a, b))
    n = len(a)
    d = (x - y) / n
    se = math.sqrt(max((x + y) / n - d * d, 0) / n)
    p = min(1.0, 2 * sum(math.comb(x + y, k) for k in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0
    return {"diff": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se, "gained": x, "lost": y, "p": p}


A, B, C = ([x["ok"] for x in res[k]] for k in ("A_human", "B_goal", "C_planner"))
tests = {"H-PL1 C vs B": mcn(C, B), "H-PL2 C vs A": mcn(C, A)}
for k, v in tests.items():
    log(k, {kk: round(vv, 4) for kk, vv in v.items()})
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731
exact = sum(norm(r["plan"]) == norm(r["instr"]) for r in test) / len(test)
by_type = {t: {k: round(sum(res[k][i]["ok"] for i, r in enumerate(test) if r["action"]["action_type"] == t) /
                     max(1, sum(r["action"]["action_type"] == t for r in test)), 3) for k in res}
           for t in sorted({r["action"]["action_type"] for r in test})}
log("planner exact match", round(exact, 3), "| per action", by_type)
json.dump({"smoke": SMOKE, "n_train": len(train), "n_test": len(test), "tests": tests, "exact_match": exact, "by_type": by_type,
           "steps": [{"ep": r["ep"], "gt": r["action"]["action_type"], "instr": r["instr"], "plan": r["plan"],
                      **{k: res[k][i] for k in res}} for i, r in enumerate(test)]},
          open(f"{W}/planner_results.json", "w"))
log("done")
