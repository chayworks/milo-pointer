"""AndroidControl (low-level instructions, test split) with Claude models via the Message Batches API, scored with the paper's strict
metric (bench/pointer_eval_local.py). Input matches what the 149M pointer reads: instruction, history/consequence string, goal, and the
first 220 accessibility-tree elements rendered with the pointer's el_text(). Labels are AndroidControl's human demonstrations.
Usage: python bench/ac_frontier.py <model> <sample.json> [limit]      (sample.json: list of test row indices, see --sample)
       python bench/ac_frontier.py --sample N seed out.json            (random test steps, fixed seed)"""

import gzip
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training/ac/androidcontrol_text.jsonl.gz"
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
PRICE = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5-5": (3.0, 15.0),
    "claude-opus-5-5": (5.0, 25.0),
}
SYSTEM = (
    "You control an Android phone for a user. You get the user's overall goal, the instruction for this one step, what happened "
    "on the last step, and the screen as a numbered list of accessibility elements (class, text, resource words, @row,col position "
    "on a 10x10 grid, flags). Choose exactly one action for this step.\n"
    'Actions: click, long_press (give "element": number), input_text (give "text": the exact text to type; the field is already '
    "focused or will be), scroll_up, scroll_down, scroll_left, scroll_right (named by where new content comes from, opposite to the finger: "
    '"swipe up" = scroll_down, "swipe down" = scroll_up, "swipe left" = scroll_right, "swipe right" = scroll_left), navigate_back, navigate_home, open_app (give "app_name"), wait (when the screen is still '
    "loading or the last action has not taken effect).\n"
    'Reply with one JSON object only and nothing else, e.g. {"action_type": "click", "element": 12}.'
)


def words(rid):
    return " ".join(re.sub(r"([a-z])([A-Z])", r"\1 \2", rid).replace("_", " ").split()).lower()


def el_text(e, W, H):  # identical to training/ac/pointer_v2/src/v2.py
    main = (e["text"] or e["desc"] or e["hint"])[:80]
    extra = [words(e["rid"])] if e["rid"] else []
    if e["desc"] and e["text"]:
        extra.append(e["desc"][:40])
    if e["hint"] and (e["text"] or e["desc"]):
        extra.append("hint " + e["hint"][:30])
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
    return f"{e['cls']} {main} | {' | '.join(extra)} @{int(y*10)},{int(x*10)} {' '.join(flags)}"


def gold_box(r):
    a = r["action"]
    if a["action_type"] not in ("click", "long_press"):
        return None
    inside = [e for e in r["els"] if e["b"][0] <= a["x"] <= e["b"][2] and e["b"][1] <= a["y"] <= e["b"][3]]
    return (
        min(inside, key=lambda e: (e["b"][2] - e["b"][0]) * (e["b"][3] - e["b"][1]))["b"] if inside else None
    )


def centre_in(e, b):
    cx = (e["b"][0] + e["b"][2]) / 2
    cy = (e["b"][1] + e["b"][3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip(" .\"'")


def gold_type(a):
    return "scroll_" + a["direction"] if a["action_type"] == "scroll" else a["action_type"]


def strict_ok(r, p):  # bench/pointer_eval_local.py strict rule
    a = r["action"]
    gt = gold_type(a)
    if p.get("action_type") != gt:
        return False
    if gt in ("click", "long_press"):
        gb = gold_box(r)
        k = p.get("element")
        return (
            bool(gb)
            and isinstance(k, int)
            and 0 <= k < min(len(r["els"]), 220)
            and centre_in(r["els"][k], gb)
        )
    if gt == "input_text":
        return norm(p.get("text")) == norm(a.get("text"))
    if gt == "open_app":
        x, y = norm(p.get("app_name")), norm(a.get("app_name"))
        return x == y or (len(x) >= 3 and (x in y or y in x))
    return True


def test_rows():
    rows = [json.loads(l) for l in gzip.open(DATA, "rt")]

    def sig(r):
        return Counter((e["text"] or e["desc"] or e["rid"], e["cls"]) for e in r["els"])

    prev = None
    for r in rows:  # identical history/consequence string to v2.py (computed over the whole file in order)
        s = sig(r)
        if prev and prev[0] == r["ep"] and prev[1] == r["instr"].strip():
            j = sum((s & prev[3]).values()) / max(sum((s | prev[3]).values()), 1)
            r["hist"] = (
                f"repeat of last step (last action {prev[2]}), screen {int(round(j * 10)) * 10}% unchanged"
            )
        else:
            r["hist"] = "new step" + (f" (last action {prev[2]})" if prev and prev[0] == r["ep"] else "")
        prev = (r["ep"], r["instr"].strip(), r["action"]["action_type"], s)
    return [r for r in rows if r["split"] == "test"]


def user_msg(r):
    els = "\n".join(f"[{k}] {el_text(e, r['w'], r['h'])}" for k, e in enumerate(r["els"][:220]))
    return f"Goal: {r['goal']}\nInstruction: {r['instr']}\nLast step: {r['hist']}\nScreen:\n{els}"


def parse(txt):
    m = re.search(r"\{.*?\}", txt or "", re.S)
    try:
        p = json.loads(m.group(0)) if m else {}
    except Exception:
        p = {}
    if isinstance(p.get("element"), str) and p["element"].isdigit():
        p["element"] = int(p["element"])
    return p


if __name__ == "__main__":
    if sys.argv[1] == "--sample":
        n, seed, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        N = sum(1 for l in gzip.open(DATA, "rt") if '"split": "test"' in l)
        idx = sorted(random.Random(seed).sample(range(N), n))
        json.dump(idx, open(out, "w"))
        print(N, "test steps; sampled", n)
        sys.exit()
    model, sample = sys.argv[1], sys.argv[2]
    idx = json.load(open(ROOT / sample))
    idx = idx[: int(sys.argv[3])] if len(sys.argv) > 3 else idx
    T = test_rows()
    rows = [T[i] for i in idx]
    client = __import__("anthropic").Anthropic(
        api_key=open(os.path.expanduser("~/.config/milo/anthropic_key")).read().strip()
    )
    think = "sonnet" in model or "opus" in model
    reqs = [
        {
            "custom_id": f"r{k}",
            "params": {
                "model": model,
                "max_tokens": 1024 if think else 400,
                "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user_msg(r)}],
                **({"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}} if think else {}),
            },
        }
        for k, r in enumerate(rows)
    ]
    batch = (
        client.messages.batches.retrieve(os.environ["FB_FROM"])
        if os.environ.get("FB_FROM")
        else client.messages.batches.create(requests=reqs)
    )
    print("batch", batch.id, len(reqs), flush=True)
    while client.messages.batches.retrieve(batch.id).processing_status != "ended":
        time.sleep(30)
    out, tin, tout = {}, 0, 0
    for res in client.messages.batches.results(batch.id):
        if res.result.type == "succeeded":
            m = res.result.message
            out[res.custom_id] = " ".join(b.text for b in m.content if getattr(b, "type", "") == "text")
            u = m.usage
            tin += (
                (u.input_tokens or 0)
                + (u.cache_read_input_tokens or 0)
                + (u.cache_creation_input_tokens or 0)
            )
            tout += u.output_tokens or 0
    res = []
    for k, (i, r) in enumerate(zip(idx, rows)):
        p = parse(out.get(f"r{k}", ""))
        res.append({"i": i, "ep": r["ep"], "gt": gold_type(r["action"]), "got": p, "ok": strict_ok(r, p)})
    pi, po = PRICE[model]
    cost = (tin * pi + tout * po) / 1e6 * 0.5
    tag = Path(sample).stem + (f"_{sys.argv[3]}" if len(sys.argv) > 3 else "")
    (ROOT / f"bench/results/ac_{model}_{tag}.json").write_text(
        json.dumps(
            {
                "model": model,
                "n": len(res),
                "ok": sum(x["ok"] for x in res),
                "tokens_in": tin,
                "tokens_out": tout,
                "est_cost_usd": round(cost, 4),
                "rows": res,
            },
            indent=0,
        )
    )
    led = ROOT / "bench/results/frontier_spend.json"
    L = json.loads(led.read_text())
    L.append(
        {
            "model": model,
            "eval": f"androidcontrol:{tag}",
            "n": len(res),
            "usd": round(cost, 4),
            "t": time.strftime("%Y-%m-%d %H:%M"),
        }
    )
    led.write_text(json.dumps(L, indent=1))
    print(
        json.dumps({"model": model, "n": len(res), "ok": sum(x["ok"] for x in res), "usd": round(cost, 4)}),
        "total spend $%.2f" % sum(x["usd"] for x in L),
    )
