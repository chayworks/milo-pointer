"""AC-GoalFront (paper/PREREGISTRATION.md): build the goal-only frontier prompts for the AC-Planner2 dev/test steps.
Claude sees what the planner sees (goal, the episode's last eight recorded actions, the screen as in ac_frontier.py: first 220
elements via el_text) plus 16 retrieved TRAIN examples (TF-IDF over goal + previous actions, episodes with ep % 20 != 0 only) whose
correct action is shown with the target element in words. No step instruction anywhere.
Usage: python bench/ac_goalfront_build.py out.jsonl   (writes one prompt per dev and test step, in planner2_results.json order)"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ac_fewshot import Index, gold_json  # noqa: E402
from ac_frontier import DATA, el_text  # noqa: E402

K, N_PREV = 16, 8
SYSTEM = (
    "You control an Android phone for a user. You get the user's overall goal, the actions already taken in this episode (oldest "
    "first), and the current screen as a numbered list of accessibility elements (class, text, resource words, @row,col position on "
    "a 10x10 grid, flags). Choose exactly one action: the next step toward the goal.\n"
    'Actions: click, long_press (give "element": number), input_text (give "text": the exact text to type; the field is already '
    "focused or will be), scroll_up, scroll_down, scroll_left, scroll_right (named by where new content comes from, opposite to the "
    'finger: "swipe up" = scroll_down, "swipe down" = scroll_up, "swipe left" = scroll_right, "swipe right" = scroll_left), '
    'navigate_back, navigate_home, open_app (give "app_name"), wait (when the screen is still loading or the last action has not '
    "taken effect).\n"
    'Reply with one JSON object only and nothing else, e.g. {"action_type": "click", "element": 12}.'
)


def act_str(r):  # same wording as the planner's "Previous actions" (training/kaggle_planner2/planner2.py)
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


def main(out):
    pool, want, prev_ep, acts = [], [], None, []
    for line in gzip.open(DATA, "rt"):
        r = json.loads(line)
        if r["split"] != "train":
            continue
        if r["ep"] != prev_ep:
            prev_ep, acts = r["ep"], []
        r["prev"] = " ; ".join(acts[-N_PREV:]) or "none"
        acts.append(act_str(r))
        if r["ep"] % 20 != 0:
            pool.append((r["goal"] + " " + r["prev"],
                         f"Goal: {r['goal']}\nPrevious actions: {r['prev']}\nCorrect action: {gold_json(r)}"))
        elif r["ep"] % 40 in (0, 20):
            els = "\n".join(f"[{k}] {el_text(e, r['w'], r['h'])}" for k, e in enumerate(r["els"][:220]))
            want.append({"split": "dev" if r["ep"] % 40 == 0 else "test", "ep": r["ep"], "goal": r["goal"], "prev": r["prev"],
                         "screen": els, "row": r})
    print("pool", len(pool), "| dev", sum(w["split"] == "dev" for w in want), "| test", sum(w["split"] == "test" for w in want))
    ix = Index([d for d, _ in pool])
    with open(out, "w") as f:
        for w in want:
            ex = "\n\n".join(pool[i][1] for i in ix.top(w["goal"] + " " + w["prev"], K))
            msg = (f"{SYSTEM}\n\nExamples of correct actions from other episodes of this dataset (the element numbers differ per "
                   f"screen; the target element is shown in words):\n\n{ex}\n\n---\nNow the current step.\n"
                   f"Goal: {w['goal']}\nPrevious actions: {w['prev']}\nScreen:\n{w['screen']}")
            r = w["row"]
            f.write(json.dumps({"split": w["split"], "ep": r["ep"], "msg": msg,
                                "gold": {"action": r["action"], "els": [{"b": e["b"]} for e in r["els"]]}}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
