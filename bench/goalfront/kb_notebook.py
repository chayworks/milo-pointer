"""AC-GoalFront: write the Kaggle Benchmarks notebook that sends one slice of the goal-only prompts to a frontier model through
Kaggle's model proxy (free daily quota) and saves the raw replies. Scoring is done locally (bench/goalfront/score.py).
Usage: python bench/goalfront/kb_notebook.py <model> <split> <start> <end> <notebook_dir>"""
import json
import sys

CODE = r'''
import glob, json, time
import kaggle_benchmarks as kbench
MODEL, SPLIT, START, END = {model!r}, {split!r}, {start}, {end}
path = sorted(glob.glob("/kaggle/input/**/goalfront_prompts.jsonl", recursive=True))[0]
rows = [r for r in map(json.loads, open(path)) if r["split"] == SPLIT and START <= r["i"] < END]
print(MODEL, SPLIT, START, END, len(rows), flush=True)
out = open(f"/kaggle/working/goalfront_{{MODEL.split('/')[1].split('@')[0]}}_{{SPLIT}}_{{START}}_{{END}}.jsonl", "w")

@kbench.task(name="milo goalfront", description="goal-only next action on AndroidControl screens")
def goalfront(llm) -> None:
    t0 = time.time()
    for n, r in enumerate(rows):
        u = None
        for attempt in range(3):
            try:
                with kbench.chats.new(name=f"{{SPLIT}}-{{r['i']}}", orphan=True):  # fresh chat: no earlier steps in context
                    txt = llm.prompt(r["msg"], temperature=0, reasoning="low", extra_api_params={{"max_tokens": 2048}})
                    u = kbench.chats.get_current_chat().usage
                break
            except Exception as e:
                txt = f"ERROR {{e!r}}"[:500]
                time.sleep(5 * (attempt + 1))
        out.write(json.dumps({{"split": SPLIT, "i": r["i"], "reply": str(txt),
                              "usage": None if u is None else {{"in": u.input_tokens, "out": u.output_tokens,
                                        "nanodollars": (u.input_tokens_cost_nanodollars or 0) + (u.output_tokens_cost_nanodollars or 0)}}}}) + "\n")
        out.flush()
        if n % 50 == 0:
            print(n, f"{{time.time() - t0:.0f}}s", str(txt)[:120], flush=True)

goalfront.run(llm=kbench.llms[MODEL])
out.close()
print("done", flush=True)
'''


def main(model, split, start, end, d):
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": CODE.format(model=model, split=split, start=int(start), end=int(end))}],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    json.dump(nb, open(f"{d}/new-benchmark-task-0ffeb.ipynb", "w"))
    meta = json.load(open(f"{d}/kernel-metadata.json"))
    meta["dataset_sources"] = ["chaitanyav33/milo-goalfront-prompts"]
    json.dump(meta, open(f"{d}/kernel-metadata.json", "w"), indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:])
