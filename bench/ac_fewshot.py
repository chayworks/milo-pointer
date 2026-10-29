"""AC-Fair (paper/PREREGISTRATION.md): bench/ac_frontier.py plus 16 retrieved AndroidControl TRAIN examples per step, so Claude sees the
dataset's labelling conventions (scroll naming, when to wait, open_app names) for steps like the one at hand. Retrieval: TF-IDF cosine over
"goal + instruction", TRAIN rows only. Same SYSTEM prompt, screen rendering and strict metric as ac_frontier.py.
Usage: python bench/ac_fewshot.py <model> <sample.json> [limit]"""

import gzip
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ac_frontier import DATA, ROOT, SYSTEM, el_text, gold_box, gold_type, parse, strict_ok, user_msg

K = 16
PRICE = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
}
EFFORT = os.environ.get("EFFORT", "medium")
RUNTAG = os.environ.get("RUNTAG", "")  # TICK: rerun / perturbed arms


def tick(r):  # TICK perturbation P: clock +1 minute, battery -1 percent; nothing added, removed or reordered
    def plus1(m):
        h, mi = int(m.group(1)), int(m.group(2)) + 1
        if mi == 60:
            h, mi = h % 12 + 1, 0
        return f"{h}:{mi:02d}"

    for e in r["els"]:
        if e["rid"] == "clock":
            e["text"] = re.sub(r"\b(\d{1,2}):(\d{2})\b", plus1, e["text"] or "")
            e["desc"] = re.sub(r"\b(\d{1,2}):(\d{2})\b", plus1, e["desc"] or "")
        if e["rid"] == "battery" and e["desc"]:
            e["desc"] = re.sub(
                r"(\d+) percent", lambda m: f"{max(int(m.group(1)) - 1, 1)} percent", e["desc"]
            )
    return r


def all_rows():  # same history/consequence string as ac_frontier.test_rows, kept for every split
    rows = [json.loads(l) for l in gzip.open(DATA, "rt")]

    def sig(r):
        return Counter((e["text"] or e["desc"] or e["rid"], e["cls"]) for e in r["els"])

    prev = None
    for r in rows:
        s = sig(r)
        if prev and prev[0] == r["ep"] and prev[1] == r["instr"].strip():
            j = sum((s & prev[3]).values()) / max(sum((s | prev[3]).values()), 1)
            r["hist"] = (
                f"repeat of last step (last action {prev[2]}), screen {int(round(j * 10)) * 10}% unchanged"
            )
        else:
            r["hist"] = "new step" + (f" (last action {prev[2]})" if prev and prev[0] == r["ep"] else "")
        prev = (r["ep"], r["instr"].strip(), r["action"]["action_type"], s)
    return rows


def light_rows(
    test_idx,
):  # streaming all_rows(): TRAIN kept only as (retrieval doc, example text); full rows only for wanted TEST steps
    want = set(test_idx)
    train, test, prev, t = [], {}, None, -1

    def sig(r):
        return Counter((e["text"] or e["desc"] or e["rid"], e["cls"]) for e in r["els"])

    for l in gzip.open(DATA, "rt"):
        r = json.loads(l)
        s = sig(r)
        if prev and prev[0] == r["ep"] and prev[1] == r["instr"].strip():
            j = sum((s & prev[3]).values()) / max(sum((s | prev[3]).values()), 1)
            r["hist"] = (
                f"repeat of last step (last action {prev[2]}), screen {int(round(j * 10)) * 10}% unchanged"
            )
        else:
            r["hist"] = "new step" + (f" (last action {prev[2]})" if prev and prev[0] == r["ep"] else "")
        prev = (r["ep"], r["instr"].strip(), r["action"]["action_type"], s)
        if r["split"] == "train":
            train.append((r["goal"] + " " + r["instr"], example(r)))
        elif r["split"] == "test":
            t += 1
            if t in want:
                test[t] = r
    return train, test


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class Index:
    def __init__(self, docs):
        tf = [Counter(toks(d)) for d in docs]
        df = Counter(w for t in tf for w in t)
        self.idf = {w: math.log(len(docs) / c) for w, c in df.items()}
        self.post = defaultdict(list)
        for i, t in enumerate(tf):
            v = {w: c * self.idf[w] for w, c in t.items()}
            n = math.sqrt(sum(x * x for x in v.values())) or 1
            for w, x in v.items():
                self.post[w].append((i, x / n))

    def top(self, q, k):
        v = {w: c * self.idf.get(w, 0) for w, c in Counter(toks(q)).items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1
        sc = defaultdict(float)
        for w, x in v.items():
            for i, y in self.post.get(w, ()):
                sc[i] += x / n * y
        return sorted(sc, key=lambda i: (-sc[i], i))[:k]


def gold_json(r):
    a = r["action"]
    t = gold_type(a)
    d = {"action_type": t}
    if t == "input_text":
        d["text"] = a.get("text", "")
    if t == "open_app":
        d["app_name"] = a.get("app_name", "")
    s = json.dumps(d)
    if t in ("click", "long_press"):
        gb = gold_box(r)
        el = next((e for e in r["els"] if e["b"] == gb), None)
        s += f"  (target element: {el_text(el, r['w'], r['h'])})" if el else "  (target: no matching element)"
    return s


def example(r):
    return f"Goal: {r['goal']}\nInstruction: {r['instr']}\nLast step: {r['hist']}\nCorrect action: {gold_json(r)}"


def fewshot_msg(r, train, ix):  # train: list of (doc, example text)
    ex = "\n\n".join(train[i][1] for i in ix.top(r["goal"] + " " + r["instr"], K))
    return (
        "Examples of correct actions from other episodes of this dataset (the element numbers differ per screen; the target "
        f"element is shown in words):\n\n{ex}\n\n---\nNow the current step.\n" + user_msg(r)
    )


if __name__ == "__main__":
    model, sample = sys.argv[1], sys.argv[2]
    idx = json.load(open(ROOT / sample))
    idx = idx[: int(sys.argv[3])] if len(sys.argv) > 3 else idx
    train, T = light_rows(idx)
    rows = [T[i] for i in idx]
    if os.environ.get("PERTURB"):
        rows = [tick(r) for r in rows]
    ix = Index([d for d, _ in train])
    msgs = [
        user_msg(r) if os.environ.get("ZS") else fewshot_msg(r, train, ix) for r in rows
    ]  # ZS: effort-confound control, no examples
    if os.environ.get("DRY"):
        print(msgs[0])
        print(sum(len(m) for m in msgs) / len(msgs), "chars/msg")
        sys.exit()
    client = __import__("anthropic").Anthropic(
        api_key=open(os.path.expanduser("~/.config/milo/anthropic_key")).read().strip()
    )
    reqs = [
        {
            "custom_id": f"r{k}",
            "params": {
                "model": model,
                "max_tokens": 8000,
                "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": m}],
                "output_config": {"effort": EFFORT},
            },
        }
        for k, m in enumerate(msgs)
    ]
    batch = (
        client.messages.batches.retrieve(os.environ["FB_FROM"])
        if os.environ.get("FB_FROM")
        else client.messages.batches.create(requests=reqs)
    )
    print("batch", batch.id, len(reqs), flush=True)
    while client.messages.batches.retrieve(batch.id).processing_status != "ended":
        time.sleep(30)
    out, tin, tout, err = {}, 0, 0, 0
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
        else:
            err += 1
    res = []
    for k, (i, r) in enumerate(zip(idx, rows)):
        p = parse(out.get(f"r{k}", ""))
        res.append({"i": i, "ep": r["ep"], "gt": gold_type(r["action"]), "got": p, "ok": strict_ok(r, p)})
    pi, po = PRICE[model]
    cost = (tin * pi + tout * po) / 1e6 * 0.5
    tag = Path(sample).stem + (f"_{sys.argv[3]}" if len(sys.argv) > 3 else "") + RUNTAG
    (ROOT / f"bench/results/acfs_{model}_{tag}.json").write_text(
        json.dumps(
            {
                "model": model,
                "n": len(res),
                "ok": sum(x["ok"] for x in res),
                "errored": err,
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
            "eval": f"androidcontrol-fewshot:{tag}",
            "n": len(res),
            "usd": round(cost, 4),
            "t": time.strftime("%Y-%m-%d %H:%M"),
        }
    )
    led.write_text(json.dumps(L, indent=1))
    print(
        json.dumps(
            {
                "model": model,
                "n": len(res),
                "ok": sum(x["ok"] for x in res),
                "errored": err,
                "usd": round(cost, 4),
            }
        ),
        "total spend $%.2f" % sum(x["usd"] for x in L),
    )
