# AC-Planner result (registered test, 2026-10-30)

Qwen2.5-1.5B-Instruct, LoRA r16, one epoch on 40,000 training steps (2.5 h on a free Kaggle T4). Executor: clean pruned retrain.
1,804 in-domain test steps (every step of episodes with ep % 40 == 20). Harness training/kaggle_planner/planner.py; per-step
results in `planner_results.json` (this folder). Registration: `paper/PREREGISTRATION.md`, section AC-Planner.

| Arm (same executor) | Accuracy | Local at tau 0.9842 | Accuracy when local |
|---|---|---|---|
| A: human step instruction | 90.2% (1628) | 72.7% | 97.0% |
| B: goal as instruction | 19.3% (349) | 25.9% | 18.4% |
| C: planner instruction | 53.0% (957) | 63.6% | 62.8% |

- H-PL1 (C > B): passes. +33.7 points [+31.1, +36.3], 688 steps gained, 80 lost, exact McNemar p < 1e-100.
- H-PL2 (C within 10 points of A): fails. C - A = -37.2 [-39.6, -34.8].
- Per action (A / B / C): click 87.7 / 19.6 / 53.4; type 96.3 / 25.2 / 62.2; open app 96.8 / 73.6 / 79.2; scroll 95.7 / 1.7 / 49.6;
  back 98.6 / 0 / 45.9; wait 85.9 / 0.8 / 25.8.
- Planner instructions match the human one exactly on 12.6% of steps; on those the executor is right 92.3%. When the planner's first
  word (the action verb) matches the human's, the executor is right 65.8%; when it does not, 28.3%.

What it means. The style problem from the pilot is solved: when the planner names the right step, the executor follows it as well as
a human instruction. What remains is planning itself: choosing the next step from the goal, four past actions and 40 screen elements.
The serious finding for the routing design is the last column: the executor's confidence measures whether it can carry out the
instruction, not whether the instruction is right, so with a planner in front, confident local steps are right only 62.8% of the time.
A goal-only system needs a separate check on the plan (planner likelihood, agreement between sampled plans, or a verifier) before
the routing result can carry over.

Next steps, in order of cost: (1) plan confidence from the planner's own token likelihood and from agreement across several sampled
instructions, measured on the ep % 40 == 0 steps, no new training; (2) a larger planner (Qwen3-4B, QLoRA) and more history; (3) the
frontier fallback for steps where the plan check fails (needs API credit).
