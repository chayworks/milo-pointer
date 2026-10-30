"""Re-score on AndroidControl-Curated labels (arXiv:2510.18488; batwBMW/AndroidControl_Curated, android_control_low_bbox.json):
action types are unchanged; a tap is correct if the centre of the chosen element lies inside the curated gt_max_bbox; typed text and
app names are compared with the curated value (same normalisation as the strict metric). Pointer: all matched test steps. Pointer,
Claude arms and the registered routers: the 1,000 fresh steps. Streams the dataset (8 GB machine).
Usage: python bench/ac_curated_score.py /path/to/android_control_low_bbox.json"""

import gzip, json, math, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from ac_router_score import THR, conf, D, strict  # registered router rule (unchanged)


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip(" .\"'")


def mcn(x, y):
    return min(1, 2 * sum(math.comb(x + y, k) for k in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0


C = {}
for x in json.load(open(sys.argv[1])):
    p = x["image"].split("/")
    C[(int(p[1]), int(p[2][5:-4]))] = x
fresh = json.load(open(ROOT / "bench/ac_fresh1000.json"))
fset = set(fresh)
LE = json.load(open(ROOT / "training/ac/pointer_v2/local_eval.json"))
TY = D["types"]
rows, boxes, t = {}, {}, -1
for l in gzip.open(ROOT / "training/ac/androidcontrol_text.jsonl.gz", "rt"):
    r = json.loads(l)
    if r["split"] != "test":
        continue
    t += 1
    c = C.get((r["ep"], r["step"]))
    if c is None:
        continue
    rows[t] = (r["action"], c)
    if t in fset:
        boxes[t] = [e["b"] for e in r["els"][:220]]


def inside(b, g):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return g[0] <= cx <= g[2] and g[1] <= cy <= g[3]


def text_ok(
    orig_ok, a, c
):  # pointer: we keep only its strict outcome, so a changed curated text counts as wrong unless it equals the original
    ct = c["gt_action"].split(":", 1)[1] if ":" in c["gt_action"] else ""
    ot = a.get("text") if a["action_type"] == "input_text" else a.get("app_name")
    return orig_ok and norm(ct) == norm(ot)


def ptr_ok(t):
    a, c = rows[t]
    le = LE[t]
    gt = le["gt"]
    if gt in ("click", "long_press"):
        return bool(le["pb"]) and inside(
            le["pb"], c["gt_max_bbox"]
        )  # pb is set only when the predicted type was right
    if gt in ("input_text", "open_app"):
        return text_ok(le["strict"], a, c)
    return le["strict"]


def claude_ok(t, got):
    a, c = rows[t]
    gt = c["gt_action"].split(":")[0]
    ty = got.get("action_type")
    want = {
        "swipe": {"up": "scroll_down", "down": "scroll_up", "left": "scroll_right", "right": "scroll_left"}
    }
    g = want["swipe"][c["gt_action"].split(":")[1]] if gt == "swipe" else gt
    if ty != g:
        return False
    if g in ("click", "long_press"):
        k = got.get("element")
        B = boxes[t]
        return isinstance(k, int) and 0 <= k < len(B) and inside(B[k], c["gt_max_bbox"])
    if g == "input_text":
        return norm(got.get("text")) == norm(c["gt_action"].split(":", 1)[1])
    if g == "open_app":
        x, y = norm(got.get("app_name")), norm(c["gt_action"].split(":", 1)[1])
        return x == y or (len(x) >= 3 and (x in y or y in x))
    return True


allt = sorted(rows)
P = {t: ptr_ok(t) for t in allt}
print(
    f"pointer, all matched test steps: curated {sum(P.values())}/{len(allt)} = {100*sum(P.values())/len(allt):.1f}%  "
    f"(original strict on the same steps {sum(LE[t]['strict'] for t in allt)})"
)
F = [t for t in fresh if t in rows]
print("fresh steps matched", len(F))
loc = {t: conf(D["res"][t]) >= THR for t in F}
for m, tag in (("opus", ""), ("sonnet", "")):
    R = {
        r["i"]: r["got"]
        for r in json.load(open(ROOT / f"bench/results/acfs_claude-{m}-5-5_ac_fresh1000.json"))["rows"]
    }
    cl = {t: claude_ok(t, R[t]) for t in F}
    cas = {t: P[t] if loc[t] else cl[t] for t in F}
    x = sum(cas[t] and not cl[t] for t in F)
    y = sum(cl[t] and not cas[t] for t in F)
    n = len(F)
    d = (x - y) / n
    se = math.sqrt((x + y) / n - d * d) / math.sqrt(n)
    print(
        f"{m}: pointer {sum(P[t] for t in F)}  {m}-FS {sum(cl.values())}  cascade {sum(cas.values())}  "
        f"cascade-{m} {100*d:+.1f} [{100*(d-1.96*se):+.1f},{100*(d+1.96*se):+.1f}] p={mcn(x,y):.2g}"
    )
    if m == "opus":
        OP = cl
    else:
        x = sum(cas[t] and not OP[t] for t in F)
        y = sum(OP[t] and not cas[t] for t in F)
        d = (x - y) / n
        se = math.sqrt((x + y) / n - d * d) / math.sqrt(n)
        print(f"  sonnet cascade - opus-FS {100*d:+.1f} [{100*(d-1.96*se):+.1f},{100*(d+1.96*se):+.1f}]")
