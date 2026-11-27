# AC-GoalFront result (registered 2026-11-25, run 2026-11-25 to 11-27)

Question: given the user's goal only (no step instruction), does routing match a frontier model acting alone? F = Claude Sonnet 5.5
on every one of the 1,804 AC-Planner test steps (ep % 40 == 20), goal-only prompt with 16 retrieved training examples
(`bench/ac_goalfront_build.py`), temperature 0, reasoning low, max 2048 tokens, a fresh chat per step, through Kaggle Benchmarks
(`kb_notebook.py`). R = the AC-Planner2 local action (4B planner + 149M executor) where its plan check trusts the plan (comb >= the
frozen dev threshold 0.6398, 905 steps), otherwise F's action (899 steps). Scored with `score.py`; per-step outcomes and replies in
`goalfront_results.json`, raw replies in `raw/`. Both slices completed: 1,804 of 1,804 replies, 0 failed calls, 0 unparsed.

- H-GF1 (R non-inferior to F, margin 3 points): PASS. R 68.1% vs F 66.5%, R - F +1.6 points [+0.1, +3.1] (paired Wald 95%).
  A 2-point margin would also pass.
- H-GF2 (R better than F; tested because H-GF1 passed): PASS. R right where F is wrong on 105 steps, the reverse on 76, exact
  McNemar p = 0.037.
- Cost: R sends 49.8% of steps to Sonnet. Sonnet cost for the test set: R $9.38, F $18.48.

| Steps | n | Local (planner + executor) | Sonnet |
|---|---|---|---|
| kept local by the plan check | 905 | 77.9% | 74.7% |
| sent to Sonnet | 899 | (not used) | 58.2% |
| all | 1,804 | 59.8% (AC-Planner2 alone) | 66.5% |

By action (R / F): click 69.5 / 69.3 (n 1,110), scroll_down 69.7 / 59.0 (195), input_text 75.6 / 73.3 (135), wait 48.4 / 54.7
(128), open_app 82.4 / 76.8 (125), navigate_back 55.4 / 50.0 (74); the other types have 17 steps or fewer.

Disclosed before the test run: 30 dev steps were sent as a pilot to check the prompt and the cost; none of the test steps was seen
before scoring. Context: the goal-only 149M pointer from E2/E14b, with no planner, is 61.7% to 63.3% on its own test sets.

What it means. With only the goal, a 4B planner and 149M executor running locally answer half the steps, and on that half they beat
Sonnet 5.5 (77.9% vs 74.7% on the same steps). Sending the other half to Sonnet gives a system that is 1.6 points more accurate than
Sonnet alone (p = 0.037) for half the Sonnet cost. The gain comes from scrolls, app launches and back presses; Sonnet is better on
waits, which the check still keeps too often. One model (Sonnet 5.5), one dataset (AndroidControl), one test set; the margin is small,
and the lower bound of the gain is +0.1 points.
