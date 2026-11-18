# AC-PlanConf result (registered 2026-11-18, run 2026-11-18)

Question: can cheap signals from the AC-Planner planner tell when its instruction is wrong? Same planner (Qwen2.5-1.5B LoRA from
AC-Planner; greedy test plans reproduce AC-Planner's on 99.4% of steps), same 149M executor, no frontier model. Signals fitted and
thresholds set on 1,691 dev steps (ep % 40 == 0); tested on the 1,804 AC-Planner test steps (ep % 40 == 20). Harness `planconf.py`,
per-step signals and outcomes in `planconf_results.json`, log `run_v2.log`. Registration: `paper/PREREGISTRATION.md`, AC-PlanConf.

Executor right on the greedy plan: 53.0% of test steps.

| Signal | AUROC | Right among kept, 30% kept | 50% kept | 70% kept |
|---|---|---|---|---|
| executor confidence (baseline) | 0.665 | 65.5% | 65.1% | 61.4% |
| plan mean token log-prob | 0.619 | 64.6% | 60.5% | 58.6% |
| plan min token log-prob | 0.654 | 66.0% | 63.5% | 61.5% |
| agreement of 8 sampled plans | 0.719 | 72.6% | 70.7% | 64.5% |
| combination (logistic, fitted on dev) | 0.747 | 76.6% | 71.2% | 64.7% |

- H-PC1 (agreement beats executor confidence): PASS. AUROC +0.054 [+0.020, +0.086], episode-cluster bootstrap, p = 0.003
  (passes under Holm).
- H-PC2 (half kept, right at least 70%): FAIL. At the dev threshold the combination keeps 47.9% of test steps; they are right
  71.2% [68.1, 74.1]. The point estimate clears 70% but the registered lower bound does not.
- Kept steps by action (right among kept / share kept): open_app 93.4% / 73%, type 86.6% / 50%, click 68.3% / 44%,
  back 61.9% / 57%.

What it means. Asking the planner several times and checking whether the executor would do the same thing is a better warning sign
than the executor's own confidence, and combining it with the planner's likelihood raises the accuracy of the kept half from 65% to
71%. That is a real but modest gain: at 50% kept, about three in ten local steps are still wrong, mostly taps where the planner picks
a plausible but wrong next step. The registered decision rule says the frontier goal-only run is not worth paying for yet; the next
step is a stronger planner (more history, a larger model), with this check reused on top.
