# AC-Planner2 result (registered 2026-11-18, run 2026-11-18/19)

Question: does a stronger planner (Qwen3-4B-Instruct-2507, QLoRA, last eight actions, step number, 60 screen elements) raise both
the executor's accuracy from the goal alone and the accuracy of the steps the AC-PlanConf check keeps? Same 149M executor, same
signals and threshold rule as AC-PlanConf, fitted on 1,691 dev steps (ep % 40 == 0), tested on the 1,804 AC-Planner test steps
(ep % 40 == 20). No frontier model. Training: all 24,000 steps in 5.1 h on one Kaggle T4 (`run_train.log`, `planner2_train.json`);
evaluation 3.1 h (`run_eval.log`). Per-step plans, samples, signals and outcomes: `planner2_results.json`. Harness `planner2.py`.

- H-P2a (more accurate than AC-Planner from the goal alone): PASS. 59.8% vs 53.0%, +6.7 points [+4.7, +8.7], 238 steps gained,
  117 lost, exact McNemar p = 1.3e-10.
- H-P2b (half kept, right at least 70%): PASS. At the dev threshold the check keeps 50.2% of test steps (905); 705 are right,
  77.9% [75.1, 80.5] (Wilson 95%; the Holm-adjusted 97.5% lower bound is 74.7%, still above 70%).
- Both pass under Holm.

| Signal | AUROC | Right among kept, 30% kept | 50% kept | 70% kept |
|---|---|---|---|---|
| executor confidence | 0.667 | 74.4% | 71.0% | 68.0% |
| plan mean token log-prob | 0.627 | 71.8% | 66.9% | 65.5% |
| plan min token log-prob | 0.669 | 75.2% | 71.9% | 68.0% |
| agreement of 8 sampled plans | 0.718 | 77.0% | 74.0% | 70.3% |
| combination (logistic, fitted on dev) | 0.747 | 80.6% | 77.9% | 72.4% |

Agreement again beats executor confidence (AUROC +0.051 [+0.016, +0.085], p = 0.006; secondary here).
Kept steps by action (right among kept / share kept): open_app 91.7% / 77%, type 88.6% / 59%, click 77.7% / 46%, scroll 77.7% / 49%,
back 75.8% / 45%, wait 50.0% / 55%. Exact match with the human instruction: 13.9% (AC-Planner 12.6%).

What it means. From the goal alone, a 4B planner plus the 149M executor answers half the steps on its own and is right on 78% of
those, against 71% for the 1.5B planner and 65% when only the executor's confidence decides. That clears the registered bar for the
next test: whether routing the other half to a frontier model, given the goal only, matches the frontier model acting alone. Waits
remain the weak spot (the check keeps half of them at 50%).
Under the registered decision rule the frontier goal-only arm is registered separately before any API call; it needs new credit.
