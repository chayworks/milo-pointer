"""Parity: the released fp16 package must reproduce the frozen pointer_v2 strict outcomes on AndroidControl test steps.
Streams the dataset and keeps only the steps it checks (the 8 GB Mac cannot hold the whole file next to the model).
Usage (repo root): .venv/bin/python release/milo-pointer/parity_check.py [n]"""

import gzip, json, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "bench"), str(Path(__file__).parent)]
from ac_frontier import DATA, strict_ok


def wanted_test_rows(want):  # same history/consequence string as ac_frontier.test_rows
    def sig(r):
        return Counter((e["text"] or e["desc"] or e["rid"], e["cls"]) for e in r["els"])

    out, prev, t = {}, None, -1
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
        if r["split"] == "test":
            t += 1
            if t in want:
                out[t] = r
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ref = [r["strict"] for r in json.load(open(ROOT / "training/ac/pointer_v2/local_eval.json"))]
    idx = json.load(open(ROOT / "bench/ac_sample1000.json"))[:n]
    T = wanted_test_rows(set(idx))
    from milo_pointer import MiloPointer

    m = MiloPointer.load(Path(__file__).parent)
    same = ok = 0
    ts = []
    for i in idx:
        r = T[i]
        t0 = time.time()
        p = m.predict(r["goal"], r["instr"], r["els"], r["w"], r["h"], r["hist"])
        ts.append(time.time() - t0)
        s = strict_ok(r, p)
        ok += s
        same += s == bool(ref[i])
    ts.sort()
    print(
        f"n={n} released {ok} vs frozen {sum(ref[i] for i in idx)}; same outcome {same}/{n}; median {1000*ts[n//2]:.0f} ms"
    )
