"""AC-Shortlist frontier arms (paper/PREREGISTRATION.md). Same system prompt, 16 TF-IDF-retrieved examples, effort and Batch API as
ac_fewshot.py; the retrieval pool excludes every validation episode (ep % 20 == 0), so no sample episode can be retrieved.
Arms:
  full       all sample steps, full screen (the frontier-alone baseline)
  full_def   deferred steps only, full screen (the baseline when only the deferred-step arms are funded)
  cand_full  deferred steps only, full screen plus the on-device model's top-5 candidates
  cand_only  deferred steps only, only the top-5 candidate elements are shown (the rest of the screen stays on the device)
Usage: python bench/sl_frontier.py <model> <arm>   (DRY=1: build prompts and count tokens, no batch)"""
import gzip
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ac_fewshot import EFFORT, K, PRICE, Index, example  # noqa: E402
from sl_prepare import val_rows  # noqa: E402
from ac_frontier import DATA, ROOT, SYSTEM, el_text, gold_type, parse, strict_ok, user_msg  # noqa: E402

INTRO = ("Examples of correct actions from other episodes of this dataset (the element numbers differ per screen; the target "
         "element is shown in words):\n\n{ex}\n\n---\nNow the current step.\n")


def train_pool():
    prev, docs = None, []

    def sig(r):
        return Counter((e["text"] or e["desc"] or e["rid"], e["cls"]) for e in r["els"])

    for line in gzip.open(DATA, "rt"):
        r = json.loads(line)
        s = sig(r)
        if prev and prev[0] == r["ep"] and prev[1] == r["instr"].strip():
            j = sum((s & prev[3]).values()) / max(sum((s | prev[3]).values()), 1)
            r["hist"] = f"repeat of last step (last action {prev[2]}), screen {int(round(j * 10)) * 10}% unchanged"
        else:
            r["hist"] = "new step" + (f" (last action {prev[2]})" if prev and prev[0] == r["ep"] else "")
        prev = (r["ep"], r["instr"].strip(), r["action"]["action_type"], s)
        if r["split"] == "train" and r["ep"] % 20 != 0:
            docs.append((r["goal"] + " " + r["instr"], example(r)))
    return docs


def step_msg(r, arm, top5):
    if arm in ("full", "full_def"):
        return user_msg(r)
    if arm == "cand_full":
        c = ", ".join(f"[{k}]" for k in top5)
        return (user_msg(r) + f"\nAn on-device model ranked these elements highest for this step, most likely first: {c}. It was "
                "not confident: the right answer may be another element or an action that needs no element.")
    els = "\n".join(f"[{k}] {el_text(r['els'][k], r['w'], r['h'])}" for k in top5)
    return (f"Goal: {r['goal']}\nInstruction: {r['instr']}\nLast step: {r['hist']}\nScreen (only the 5 elements an on-device "
            "model ranked highest for this step are shown, most likely first; the rest of the screen is hidden; the right answer "
            f"may also be an action that needs no element):\n{els}")


if __name__ == "__main__":
    model, arm = sys.argv[1], sys.argv[2]
    L = json.load(open(ROOT / "bench/results/sl_local.json"))["rows"]
    pick = [j for j, x in enumerate(L) if arm == "full" or not x["local"]]
    V = val_rows()
    rows = [V[L[j]["k"]] for j in pick]
    assert all(r["ep"] == L[j]["ep"] for j, r in zip(pick, rows))
    del V
    docs = train_pool()
    ix = Index([d for d, _ in docs])
    msgs = [INTRO.format(ex="\n\n".join(docs[i][1] for i in ix.top(r["goal"] + " " + r["instr"], K))) + step_msg(r, arm, L[j]["top5"])
            for j, r in zip(pick, rows)]
    del docs, ix
    client = __import__("anthropic").Anthropic()  # reads ANTHROPIC_API_KEY
    if os.environ.get("DRY"):
        print(msgs[0][-1500:])
        probe = msgs[:: max(1, len(msgs) // 20)]
        n_tok = [client.messages.count_tokens(model=model, system=SYSTEM, messages=[{"role": "user", "content": m}]).input_tokens
                 for m in probe]
        per = sum(n_tok) / len(n_tok)
        pi, po = PRICE[model]
        print(f"{arm}: {len(msgs)} requests, ~{per:.0f} input tokens each; est. batch cost ${len(msgs) * (per * pi + 60 * po) / 1e6 * 0.5:.2f}")
        sys.exit()
    reqs = [{"custom_id": f"r{k}", "params": {"model": model, "max_tokens": 8000,
                                               "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                                               "messages": [{"role": "user", "content": m}], "output_config": {"effort": EFFORT}}}
            for k, m in enumerate(msgs)]
    batch = client.messages.batches.create(requests=reqs)
    print("batch", batch.id, len(reqs), flush=True)
    while client.messages.batches.retrieve(batch.id).processing_status != "ended":
        time.sleep(30)
    out, tin, tout, err = {}, 0, 0, 0
    for res in client.messages.batches.results(batch.id):
        if res.result.type == "succeeded":
            m = res.result.message
            out[res.custom_id] = " ".join(b.text for b in m.content if getattr(b, "type", "") == "text")
            u = m.usage
            tin += (u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
            tout += u.output_tokens or 0
        else:
            err += 1
    res = []
    for k, (j, r) in enumerate(zip(pick, rows)):
        p = parse(out.get(f"r{k}", ""))
        res.append({"j": j, "ep": r["ep"], "gt": gold_type(r["action"]), "got": p, "ok": strict_ok(r, p)})
    pi, po = PRICE[model]
    cost = (tin * pi + tout * po) / 1e6 * 0.5
    (ROOT / f"bench/results/sl_{model}_{arm}.json").write_text(json.dumps(
        {"model": model, "arm": arm, "batch": batch.id, "n": len(res), "ok": sum(x["ok"] for x in res), "errored": err,
         "tokens_in": tin, "tokens_out": tout, "est_cost_usd": round(cost, 4), "rows": res}, indent=0))
    led = ROOT / "bench/results/frontier_spend.json"
    lg = json.loads(led.read_text()) if led.exists() else []
    lg.append({"model": model, "eval": f"ac-shortlist:{arm}", "n": len(res), "usd": round(cost, 4), "t": time.strftime("%Y-%m-%d %H:%M")})
    led.write_text(json.dumps(lg, indent=1))
    print(json.dumps({"arm": arm, "n": len(res), "ok": sum(x["ok"] for x in res), "errored": err, "usd": round(cost, 4)}))
