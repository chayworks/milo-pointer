# Pre-registration log

How to read this log. Each section was written and committed to version control before the data it describes were scored; results
were appended afterwards under the same heading. Labels: **E1-E24** are development experiments on AndroidControl and live
AndroidWorld tasks; **W3-W12** are registered tests of the web system on fresh pages; **AC-*** are the AndroidControl comparisons
with frontier models that the paper reports (AC, AC-Opus, AC-Fair, TICK, AC-Router, AC-Router-Sonnet, AC-Router-Rerun,
AC-Router-Haiku). Entries are kept as written, including ones whose hypotheses failed; amendments are marked as such. Model and
system names in early entries (Milo, pointer, ScreenQL) refer to the systems described in the paper.

# Pre-registration of pending experiments (committed before any result exists)

All on the AndroidControl test split, strict metric unless stated, with 95% episode-bootstrap CIs (`bench/ac_stats.py`).

## E1: Point vs generate (controlled)
- **Setup:** Qwen2.5-0.5B-Instruct (and 1.5B when GPU allows), LoRA r=32, generating the action string from *exactly* the
  Milo Pointer v2 input (same element listing, history/consequence line, train split), ≤ 8.5 h on a T4.
  Compared with Milo Pointer v2 (149M).
- **H1:** the 149M pointer ≥ the 0.5B generator on overall strict accuracy.
- **Decision rule:** "pointing matches or beats generating at 3.4× fewer parameters" is claimed only if the paired-bootstrap
  difference (pointer − generator) has a 95% CI lower bound ≥ −1.0 point. Otherwise we report that generating wins.
- **Confound, disclosed:** the generator may see fewer training examples in 8.5 h; we report examples seen for both.

## E2: Goal-only (high-level) vs LiMAC
- **Setup:** Milo Pointer (149M) trained with the goal + gold action history only (no step instruction). Strict metric.
- **Comparison:** LiMAC reports 63.1% overall accuracy (bbox rule, goal-only).
- **H2:** pointer > 63.1% (overall). A claim of beating LiMAC requires the 95% CI lower bound > 63.1.
- **Disclosed differences:** LiMAC's exact test subset and action-matching details may differ; this is noted next to the result.

## E3: Scale (ModernBERT-large, 395M) and features (v4: section context + list order)
- **H3a:** large > v2 (paired CI lower bound > 0). **H3b:** v4 > v2 (paired CI lower bound > 0).
- **Confound, disclosed:** large runs 1 epoch at 2,048 tokens (time limit) vs v2's ~1.9 epochs at 3,072.
- **Headline rule:** we claim "matches UI-TARS-2B" only if our OS-Atlas-protocol SR is ≥ 89.3 with a CI lower bound ≥ 88.3.

## E4: Seeds
- The headline pointer is re-trained with ≥ 2 more seeds when GPU quota allows. The paper reports mean ± std. If std > 1.0, the
  single-run claims are softened accordingly.

## Addendum (2026-10-01): fair re-runs once GPU quota resets (registered before running)
v3, v4 and large were all confounded by Kaggle's time limit (fewer steps, and 2,048 tokens for large). The batched trainer
(`PT_BATCH=8`, loss verified identical to per-sample) should give 2–3× throughput. The planned re-runs, all at **equal optimizer
steps** (= v2's schedule, 2 epochs, 3,072 tokens):
1. v2 seed 2 and seed 3 (E4: seed variance of the headline).
2. ModernBERT-large, 2 epochs (E3a, fair).
3. v4 (section context + list order) (E3b, fair).
The same decision rules as above apply. If throughput does not reach 2 epochs in 9.5 h, the run is reported as incomplete, not as a result.

## E5 (registered 2026-10-02, before any run): the "generation tax" across scale
**Question.** At identical input and identical training data, how large must a decoder that *generates* actions be to match a
149M encoder that *points* at elements and *copies* text spans?
- **Data:** the fixed 18,481-example AndroidControl subset (seed 7), 1 pass, the same 2,560-token input cap, the same strict metric.
- **Arms:** pointer (ModernBERT-base, 149M; seeds 0/1/2). Generators: Qwen2.5-{0.5B, 1.5B, 3B, 7B}-Instruct, LoRA r = 32,
  same prompt format (`tools/build_generator_kaggle.py`).
- **Primary outcome:** the smallest generator size whose paired-bootstrap 95% CI vs the pointer includes 0. If none does up to 7B,
  report "> 47×".
- **Secondary:** a per-action-type breakdown (typed text / app name, i.e. copy vs generate; clicks, i.e. point vs generate).
- **Both outcomes are reported:** (a) no generator ≤ 7B matches → "output structure is worth > 47× parameters"; (b) a generator matches
  at size X → a "generation tax ≈ X / 149M" scaling estimate.
- **Generality check (E6):** repeat the 149M pointer vs the 0.5B generator on a second domain where actions copy from the
  input (a function-calling benchmark with argument spans), same rules.
- **Prior art cited:** pointer-generator networks (See et al. 2017) and copy mechanisms. The contribution is the controlled scale measurement
  for agent actions, not the mechanism itself.

### E5 amendment (2026-10-02, before the 3B / 7B runs start)
- For budget reasons the 3B and 7B generators are evaluated on a **fixed random 4,000-step test subset** (seed 11, over the 8,444 test steps).
  All other arms (the pointer seeds, 0.5B, 1.5B) are re-scored on the **same** subset for paired comparisons; full-test numbers stay reported too.
- The 3B and 7B base weights are loaded in bf16 (a 7B model in fp32 does not fit a 40 GB A100). The 0.5B and 1.5B used fp32 master weights with
  fp16/bf16 autocast. Disclosed as a recipe difference.
- Training is one full pass over the identical 18,481-example subset. A run that cannot complete the pass is reported as incomplete, not as a result.

## E7 (registered 2026-10-02, before any run): confidence-routed cascade, 149M pointer → 7B generator
- On the fixed 4,000-step eval subset (seed 11), for each step: if the pointer's confidence (type probability × top element probability)
  is ≥ τ, use the pointer; otherwise use the 7B generator's prediction (from E5).
- **τ is chosen on held-out dev steps, never on test.** Fallback if no dev predictions exist: τ = the pointer's 80th-percentile confidence
  on the *training* subset (so the pointer handles ~80% of steps).
- **Report:** cascade accuracy with 95% CI, paired vs the pointer alone and vs the 7B alone, plus the fraction of steps sent to the 7B (≈ cost).
- **Claim rule:** "cascade beats both" only if both paired CI lower bounds are > 0.

## E8 (registered 2026-10-02, before any run): layout-aware pointer
- Same as Pointer v2 (input, data, steps, batched trainer), plus learned 2D box embeddings: (x0, y0, x1, y1, w, h) normalised to the
  screen and bucketed into 0–31, summed into each element's marker embedding.
- **Primary:** strict accuracy vs v2 at an equal step count (paired CI). **Secondary:** accuracy on steps whose instruction contains
  spatial words (first / second / last / top / bottom / left / right / next to / below / above). The word list is fixed here.
- **Prior art cited:** LayoutLM, UIBert, Spotlight.

## E9 status (2026-10-02): latent probe vs generation — prior art found; demoted to supporting analysis
A prior-art search found the core idea already shown: an attention selector beats LLMs' own *tool* choices across 0.5B–32B (arXiv 2606.16364), and frozen
MLLM attention beats generated coordinates for GUI grounding (TAG, arXiv 2412.10840). An E9 run (`tools/latent_probe.py`) would only extend this to
accessibility-tree element choice with a scale sweep and a pointer baseline. It is **not** claimed as novel; it is run only if budget remains after E5.

## E10 (registered 2026-10-02, before analysis): certified selective action, with a user-chosen bound on the WRONG-action rate
**Gap (prior-art check):** CORA (2604.09155) bounds *harmful* actions on its own Phone-Harm set with a VLM guardian; SafeGround bounds grounding only;
Argus (2606.25760) warns coverage degrades under interface shift. No work bounds the wrong-action rate of the whole action (type + element + copied
text) on public splits, with an exact finite candidate set (pointer), under unseen-app shift.
**Method:** Learn-Then-Test (Angelopoulos et al. 2021). Score = P(type) × P(element) (for clicks) from Pointer v2. Thresholds λ are scanned from
high to low with fixed-sequence testing and a binomial-tail p-value of H0: selective error > α; choose the lowest λ whose test rejects at δ.
Guarantee (exchangeable data): P(error among acted steps ≤ α) ≥ 1 − δ. The agent acts if score ≥ λ̂, otherwise asks the user.
**Settings:**
- A, exchangeable: IDD episodes randomly halved into calibration and test (200 random splits).
- B, shift: calibrate on all IDD, test on app / task / category unseen.
- C, strict app holdout (appdev pointer): calibrate on seen-app episodes, test on never-seen apps.
**Report:** α ∈ {1%, 2%, 5%}, δ = 0.1. Realised selective error, coverage (fraction acted), and the violation frequency over the 200 splits (it must be ≤ δ in A).
**Hypotheses:** H10a: the guarantee holds in A. H10b: it is violated under shift in B or C (Argus's warning). H10c: calibrating on held-out *apps*
(group / leave-apps-out calibration) restores validity in C. **No claim beyond what the numbers show.**

### E10 result record + H10d (registered 2026-10-02 ~09:00, after H10c's result, BEFORE H10d is run)
- C (calibrate seen-app episodes → test never-seen apps, appdev pointer): α=5%: λ̂=0.972, coverage 69.4%, selective error 4.99%
  (app-cluster bootstrap 95% CI [3.88, 6.77]) — not violated, but borderline. α=1%, 2%: nothing certified (always ask).
- H10c (leave-apps-out, 72 never-seen apps halved, 200 splits): α=5%: violation rate 18.5% > δ=10% → **H10c NOT supported**.
  Mean coverage 54.7%, mean selective error 3.65%.
- Diagnosis (hypothesis, not yet tested): step-level LTT assumes exchangeable steps; steps cluster within apps, so with ~36 calibration
  apps the binomial p-value is anti-conservative at the app level.
- **H10d:** an app-level LTT restores validity in the leave-apps-out protocol. Per calibration app a and threshold λ:
  X_a(λ) = (wrong_a(λ) − α·acted_a(λ)) / n_a ∈ [−α, 1−α] (n_a = steps of app a). Target: E_app[X_a(λ)] ≤ 0 (expected excess wrong actions per
  step, for a new app, ≤ 0). p-value: Hoeffding, p = exp(−2 m (−mean X)^2 / 1²) when mean X < 0 else 1, m = #calibration apps; Bonferroni over
  the same fixed grid; δ = 0.1. Evaluated with the same 200 leave-apps-out splits; the test metric is the app-averaged excess on the held-out
  half (violation = mean X > 0) and, secondarily, the pooled selective error. Expected cost: much lower coverage. Reported whatever it shows.
- **H10d result:** app-level Hoeffding LTT has 0% violations but certifies nothing (coverage 0 at every α). This was knowable in advance and we
  missed it: Bonferroni level δ/153 with m≈36 apps needs mean excess ≤ −sqrt(ln(153/0.1)/(2·36)) ≈ −0.32, but excess ≥ −α. Hoeffding at α=5%
  would need roughly 1,500 calibration apps. For reference, step-level LTT's app-averaged violation rate in the same splits is 29% (pooled 18.5%).
- **H10e (registered before running):** same app-level target and splits, but a variance-adaptive bound: Waudby-Smith–Ramdas betting
  (hedged capital process, as used for RCPS in Bates et al. 2021), on X_a shifted to [0,1] via (X_a + α), testing H0: E[X_a + α] ≥ α;
  Bonferroni over the grid; δ = 0.1. H10e: violations ≤ δ AND mean coverage > 0 at α = 5%. Every attempt (H10c, H10d, H10e) is reported.
- **H10e result:** WSR betting also certifies nothing (coverage 0, violations 0 at every α). This is not a weakness of the bound. It is
  impossible for any valid test, by a simple argument: the null "Y_a = 1 with probability α, else 0" has mean exactly α, and under it a
  calibration set of m perfectly clean apps occurs with probability (1−α)^m. Any level-δ test that certifies on the best possible data
  therefore needs (1−α)^m ≤ δ, i.e. m ≥ ln(1/δ)/(−ln(1−α)) calibration APPS. At δ=0.1: α=5% → 45 apps, 2% → 114, 1% → 229 (with Bonferroni over
  our 153-point grid: 143 / 363 / 730). Our leave-apps-out splits have 36 calibration apps, so no valid app-level certificate is possible.
- **E10 conclusion (what the numbers show):** (1) step-level LTT is valid when steps are exchangeable (A) but anti-conservative under app
  clustering: 18.5% pooled / 29% app-averaged violations at nominal δ=10% (H10c). (2) Valid app-level certificates exist, but they are
  limited by the number of distinct calibration apps, not steps: ≥45 apps for α=5%, δ=0.1. AndroidControl's never-seen-app split has 72.
  This gives a concrete data requirement for certifying GUI agents on new apps.

## E12 (registered 2026-10-02 ~09:40, before any run): time-to-trust — earning autonomy app by app, online, with an anytime-valid guarantee
**Motivation (from E10):** offline certificates fail on new apps; any valid app-level certificate needs ≥45 calibration apps (α=5%, δ=0.1).
**Protocol:** appdev pointer (never trained on these apps), its 72 never-seen apps, each app's steps streamed in episode order (the order a
user would produce them). Per app, the agent starts with no trust. Fixed threshold grid G = {0.99, 0.95, 0.90} on score = P(type)·P(element).
At each step with score s, the agent ACTS alone if some λ ∈ G is certified and s ≥ min certified λ; otherwise it ASKS (shows its proposal; the user confirms or corrects).
**Feedback models:** F1 (correction): asked steps are labelled, and a wrong autonomous action is noticed and corrected by the user (so labelled).
F2 (audit): asked steps are labelled; autonomous steps are labelled only through random audits with probability p = 0.2, using inverse-probability weighting.
**Certification (anytime-valid):** for each λ, an e-process over the labelled steps with s ≥ λ, testing H0: P(wrong | s ≥ λ, past) ≥ α:
K_t(λ) = Π (1 + b·(α − w_i)) (F1), or Π (1 + b·(α − w_i·A_i/p)) with b ≤ p/(1 − α·p) ... capped for nonnegativity (F2), with b = 0.5 / (max loss − α).
λ is certified once K_t(λ) ≥ |G|/δ (Ville + Bonferroni). **Revocation:** a second e-process on acted steps tests H0': error ≤ α and revokes trust in
that app when it exceeds |G|/δ. δ = 0.1, α ∈ {5%, 10%}.
**Report:** time-to-trust per app (steps until first certification; censored if never), fraction of apps that earn trust within their data,
overall autonomy (fraction of steps acted alone), user burden (asks), realised wrong-action rate on autonomous steps (pooled; per app; fraction of
apps with ≥20 autonomous steps whose rate exceeds α). Baselines: always ask; offline seen-app LTT threshold from setting C (autonomy from step 1, no guarantee on new apps).
**Lower bound (stated in advance):** under F1 with this grid, certification needs at least ln(|G|/δ)/ln(1 + b·α) clean steps above λ
(b as above); we report the measured time-to-trust next to it.
**Hypotheses:** H12a: realised autonomous error ≤ α pooled, and the fraction of apps violating is small (reported, no fixed cut).
H12b: the agent earns trust in a substantial fraction of apps; time-to-trust is reported as a distribution. **No claim beyond the numbers.**
- **E12 result (as registered):** α=10%: the offline seen-app threshold reaches 12.3% error on never-seen apps (violates α). Online time-to-trust:
  8/72 apps earn trust, median 131 steps (IQR 111–180); autonomous error 4.3% pooled, 0/8 apps above α, 0 revocations; overall autonomy 31.8%
  (F1) / 30.9% (F2, 20% audits). α=5%: 2/72 apps earn trust (429, 757 steps); autonomous error 2.2%; autonomy 7.9%. Valid but slow: most apps
  have fewer steps than the time-to-trust. Lower bound with this grid and bet: 63 (α=10%) / 131 (α=5%) clean steps.
- **E12b (registered before running): "offline proposes, online certifies".** (i) One candidate per app: λ0 = the seen-app LTT threshold at α/2
  (computed only from seen apps, so it is fixed before any new-app data), |G| = 1. (ii) Adaptive bet: b_t = min(0.9/(maxL − α), sqrt(2 ln(1/δ) /
  (n_t σ̂²_t))) with σ̂² the predictable WSR-style running variance of the loss (prior 1/4 weight 1), n_t = labelled steps above λ0 so far + 1.
  Revocation as before. Same metrics; F1 and F2; α ∈ {5%, 10%}. Lower bound with |G|=1, b → 1/(1−α): ln(1/δ)/(−ln(1−α)) = 45 / 22 steps.
- **E12b result:** α=10% (λ0 = 0.972 from seen apps at α/2): 16/72 never-seen apps earn trust, median time-to-trust **34 steps** (IQR 32–44;
  bound 22), vs 131 in E12. Autonomous error 3.8% pooled; autonomy 34.7% of all steps. 1 of 9 apps with ≥20 autonomous steps has a realised
  rate above α (desygner 5/43 = 11.6%), but its error above λ0 over its whole stream is 7.4% < α, so this is finite-sample noise. **0/16 certified
  apps have a true (whole-stream) error above α.** α=5% (λ0 = 0.9991): 6 apps, median 228 steps, autonomous error 1.2%, autonomy 7.2%.
  F1 and F2 are identical here: labels on autonomous steps only feed revocation, which never fired. Offline C at α=10% for comparison:
  12.3% error (violated).

## E12c (registered before running): replication of E12b on the official split
Pointer v2 (trained on the full AndroidControl train split). λ0 = LTT threshold at α/2 calibrated on IDD test episodes (apps seen in training).
Streams: each app of the official app_unseen test episodes (app = most frequent non-system package), steps in dump order. Same E12b procedure
(one candidate λ0, adaptive bet, revocation, δ = 0.1, F1), α ∈ {5%, 10%}. Report the same metrics, plus each certified app's whole-stream error
above λ0 (false-certification count). Also report the offline baseline: the IDD threshold at α applied to app_unseen.
- **E12c result (replication, official app_unseen: 66 apps, 3,449 steps; pointer v2; IDD calibration 3,897 steps):** offline IDD threshold
  violates on new apps at BOTH α: 5.35% error at α=5% (71% autonomy) and 10.58% at α=10%. Online: α=10%, λ0 = 0.983, 13/66 apps earn trust,
  median time-to-trust 38 steps (IQR 31–64), autonomous error 4.1%, 0/8 realised violations, **0/13 false certifications**, no revocations;
  autonomy 15.7%. α=5%: the IDD calibration certifies no λ0 at α/2 (handled as "never trusted", which the protocol did not spell out;
  disclosed), so 0 autonomy, 0 errors.

## E13 (registered 2026-10-02 ~10:10, before any AndroidWorld run): split-brain agent on live AndroidWorld
**Agents (same LLM in both):** (A) T3A from android_world, unmodified: the LLM sees the full a11y element list and picks actions by index.
(B) SplitAgent (tools/aw_split_agent.py): the LLM sees a compressed screen (visible texts, no indices) and emits 1-5 step instructions.
Milo Pointer v2 (149M, trained on AndroidControl only, never on AndroidWorld) executes them. Re-plans at plan end, on an unchanged screen,
or when executor confidence < τ. τ ∈ {0, 0.5, 0.9}.
**LLM:** Claude (Haiku 4.5 primary; Sonnet 5.5 if budget allows) or Gemini Flash, whichever key is available. Logged per run.
**Tasks:** all 116 AndroidWorld tasks, seed 30, 1 trial each (the paper's default), the runner's default step limits.
**Primary metric:** task success from AndroidWorld's own checkers. **Cost:** LLM calls, input+output tokens and LLM seconds per task.
**Latency:** executor seconds per step.
**Hypotheses:** H13a: success(B, best τ) ≥ success(A) − 3 points. H13b: B uses ≥ 5× fewer LLM input tokens per task than A. Both reported
whatever the outcome; the success-vs-tokens curve over τ is reported in full. Per-task paired differences use a McNemar test.
**Disclosed confounds:** domain shift (the executor never saw AndroidWorld apps); the planner prompt was written by us and is fixed before
the first run (no tuning on AndroidWorld results; any later prompt change is a new, separately reported run).
- **E13 amendment (before any run):** LLM fixed to **gemini-3.5-flash** (AI Studio) for both agents (T3A and SplitAgent). gemini-3.8-flash
  returned 503 (overloaded) at setup, so it was not used. Default generation settings in both agents.

## E13p (registered before running): offline pilot of the split on AndroidControl high-level steps
300 test steps, sampled with random.Random(13) from the 8,444 test steps. Goal + gold history of previous actions, as in AndroidControl high-level.
LLM gemini-3.5-flash for both arms. (G) Gemini alone: sees goal, history and the full element list (the pointer's own element text,
'#i' indices); returns a JSON action. (S) Split: Gemini sees goal, history and the compressed screen (visible texts only, no indices);
returns ONE step instruction; Pointer v2 executes it. Strict metric for both (click centre in gold element; exact typed text; scroll direction;
open_app containment; other types by type). Report accuracy with paired episode-bootstrap CI of S − G, Gemini input tokens per step for
each arm, and the reference goal-only pointer (E2) on the same steps. H13p: S ≥ G − 3 points. Per-step offline evaluation calls the
planner at every step, so this pilot measures accuracy, not the live cost saving.
- **E13/E13p amendment (before any data):** gemini-3.5-flash is called through **Vertex AI** (location `global`, user's billed GCP project).
  The AI Studio free tier caps it at 20 requests/day. Both agents use temperature 0 and top_p 0.95 (AndroidWorld's GeminiGcpWrapper defaults).
  T3A's prompts and logic are unchanged; only the transport is a Vertex wrapper with the same retry pattern. The 8 E13p steps collected
  earlier via AI Studio (no temperature set) are discarded and the pilot restarts from scratch.
- **E13 smoke-test bug fix (before the full run):** in the first 3-task smoke run, SplitAgent's "screen unchanged" check ignored focus,
  checked and selected changes. Tapping a text field therefore looked like no change, which triggered a needless re-plan and a duplicate tap
  (4 of 12 steps wasted in ContactsAddContact). Fixed: those flags count as a change, and a re-plan happens only if the same instruction
  leaves the screen unchanged twice. Planner prompt unchanged. Smoke runs are debugging only and are not reported as results.
- **E13 amendment after the smoke tests (3 tasks, debugging only; before the full run):**
  (i) Escalation semantics: when executor confidence for a tap < τ, the LLM grounds THAT step from the full indexed element list (T3A-style
  but for one instruction) instead of re-planning. Re-planning could not fix a low-confidence tap, because the executor would repeat it.
  τ ∈ {0 (never escalate), 0.5, 0.9} as registered; the extra tokens count toward the split agent's cost.
  (ii) Typed text: when the copied span lies inside a quoted string of the instruction (the planner prompt asks for quotes), the whole
  quoted string is typed. The smoke run had lost a leading "+" in a phone number.
  Neither change is specific to a task; both are reported. Planner prompt unchanged.
- **E13 amendment 2 (smoke debugging, before the full run):** (i) the quoted-text rule strips stray quote characters from the copied span
  (the smoke run typed "'+139…"). (ii) The planner's compressed screen now shows element state: [on]/[off] for checkable elements, [focused]
  for the focused field, and "label: content" for filled fields. Without state the planner looped (it could not see whether Bluetooth was
  off). It is still text-only, with no indices or coordinates. The 3 smoke tasks (ContactsAddContact, SystemBluetoothTurnOff,
  ClockStopWatchRunning) were used for debugging, and this is disclosed. In the full run they are reported like every other task.
- **E13 code FROZEN for the full run at commit a85de3f** (tools/aw_run.py, aw_split_agent.py, aw_bridge.py). Last smoke (debugging,
  not a result): split τ=0.9 2/3 (stopwatch, bluetooth); T3A 3/3. Gemini input tokens per task: split 1.1k–2.1k, T3A 38k–174k.
  Full runs: T3A (host 2), split τ=0.9 (host 1), then split τ=0; all 116 tasks, seed 30, 1 combination each.
- **E13p result (300 AndroidControl high-level test steps, gemini-3.5-flash via Vertex): H13p NOT supported.** Gemini alone 61.3%, split
  49.0%, S − G = −12.3 [−18.5, −6.3] (episode bootstrap). Gemini input tokens per step: 1,956 vs 347 (5.6× fewer). By gold type:
  click 69.1 vs 65.4 (near parity: the 149M executor grounds about as well as Gemini); input_text 66.7 vs 66.7; scroll 42.2 vs 15.6;
  open_app 77.8 vs 16.7; wait 36.8 vs 0.0. The loss is in the instruction choice (the planner sees only a compressed screen and rarely says
  "open app"/"wait"; scroll-direction conventions differ), not in grounding. Steps right only for G: 58; right only for S: 21.

## E14 (registered 2026-10-02 before computing): confidence cascade, 149M goal-only model first and Gemini only when unsure
Data already collected: (a) Gemini-alone (gemini-3.5-flash, E13p arm G) per-step correctness on the 300 E13p steps; (b) the E2 goal-only
pointer (149M, AndroidControl high-level) per-step correctness and confidence on all 8,444 test steps. Cascade at threshold t: steps where
the pointer's confidence ≥ t use the pointer; the rest use Gemini. Thresholds: a fixed grid t ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}.
Report accuracy and the fraction of steps sent to Gemini for each t, plus pointer-alone and Gemini-alone on the same 300 steps.
**H14:** some t in the grid gives a cascade accuracy ≥ Gemini-alone while sending ≤ 50% of steps to Gemini. Because t is chosen on the same
300 steps, a confirming result is exploratory. It must be confirmed on a FRESH sample (E14b: 1,000 new test steps, Gemini-alone collected
anew, t fixed from E14 before seeing E14b).
- **E14 result (exploratory, 300 E13p steps):** pointer-alone 63.0, Gemini-alone 61.3. Cascade: t=0.5 64.0 (Gemini on 19% of steps),
  0.6 65.3 (29%), **0.7 67.7 (40%)**, 0.8 66.7 (48%), 0.9 65.0 (58%), 0.95 65.0 (69%).
- **E14b locked before any E14b data:** t = 0.7. Sample: 1,000 test steps drawn with random.Random(14) from the test steps NOT in E13p.
  Gemini-alone (same prompt, gemini-3.5-flash, Vertex, temperature 0) is collected fresh. Primary: cascade accuracy − Gemini-alone accuracy,
  paired episode-bootstrap 95% CI, and the fraction of steps sent to Gemini. H14b: CI lower bound > 0 with ≤ 50% of steps sent to Gemini.
  Secondary: pointer-alone vs Gemini-alone.

## E15 (registered before any run): live cascade on AndroidWorld — 149M goal-only model first, T3A (Gemini) when unsure
Agent `cascade_g35` (tools/aw_cascade_agent.py, tools/aw_run.py). At each step the E2 goal-only Milo Pointer (149M, trained on AndroidControl
only) proposes an action with confidence P(type)·P(element). If confidence ≥ t = 0.7 (locked from E14) and the action is not "wait", it is
executed. Otherwise that step is delegated to T3A (gemini-3.5-flash, Vertex, unmodified prompts), which also ends the episode (status/answer;
the pointer has no "done" action). 116 tasks, seed 30, 1 combination. Compared with the full T3A run (same LLM, same tasks).
Metrics: task success (AndroidWorld checkers), LLM input/output tokens and calls per task, fraction of steps taken by the pointer.
**H15:** success(cascade) ≥ success(T3A) − 3 points with ≥ 30% fewer LLM input tokens. Reported whatever the outcome; McNemar on per-task pairs.
Disclosed: the pointer has never seen AndroidWorld apps; its confidence may be miscalibrated there (also reported: pointer-step accuracy is not
observable live, so we report per-task outcomes split by the share of pointer steps).
The split τ=0 run queued on host 1 is replaced by this run (decided before any E13 full-run result was inspected beyond the first 25 tasks).
- **E14b result (confirmatory; 1,000 fresh AndroidControl high-level test steps; t = 0.7 locked before data): H14b SUPPORTED.**
  Gemini-3.5-Flash alone 57.1%; 149M goal-only pointer alone 63.3% (+6.2 [+2.6, +10.0] over Gemini); cascade 67.2%, i.e.
  **+10.1 [+7.3, +12.8] over Gemini alone, with Gemini called on only 38% of steps.** Episode-bootstrap 95% CIs.
- **E14b breakdown (descriptive):** the 149M model wins on wait / navigate_back / scroll / open_app / typing. Gemini wins on clicks (65.0 vs 61.3);
  the cascade gets 70.0 on clicks. Confident steps: 149M 75.4 vs Gemini 59.1; unsure steps: Gemini 53.8 vs 149M 43.6. **Caveat:** the 149M
  model was trained on AndroidControl and knows its labelling conventions (when "wait" is used, scroll directions); Gemini is zero-shot. Part
  of the gap is in-domain training. The live AndroidWorld test (E15) removes that advantage.

## E14c (registered before running): the same cascade against a stronger frontier model
gemini-3.1-pro-preview (Vertex, temperature 0, same prompt as E13p arm G) on the identical 1,000 E14b steps. Cascade with t = 0.7 (locked).
Report Pro alone, the 149M model alone, and the cascade (149M + Pro), with paired episode-bootstrap CIs vs Pro alone and Pro calls (%).
H14c: cascade(149M + Pro) ≥ Pro alone (CI lower bound > 0). Secondary: cascade(149M + Flash, E14b) vs Pro alone (cost-quality frontier).
- **E14b by official split (descriptive; splits overlap):** cascade − Gemini: IDD +9.7 [+5.5, +13.8]; app_unseen +9.1 [+4.2, +13.6];
  task_unseen +10.8 [+7.1, +14.7]; category_unseen +10.3 [+6.1, +14.5]. Gemini called on 36–40% of steps.

## E14d (registered before running): a stronger baseline, Gemini few-shot with the dataset's conventions
Same 1,000 E14b steps, gemini-3.5-flash, same prompt as arm G plus 8 fixed in-context examples drawn with random.Random(8) from AndroidControl
TRAIN high-level steps (each: goal, previous actions, gold action as JSON with the target element's text). One example per action type where
available (click, scroll, open_app, wait, input_text, navigate_back, long_press, navigate_home), so the conventions are visible.
H14d: cascade (149M + few-shot Gemini, t = 0.7) > few-shot Gemini alone (CI lower bound > 0). Also reported: few-shot vs zero-shot Gemini.
- **E14d result (confirmatory, 1,000 E14b steps): H14d SUPPORTED.** Few-shot gemini-3.5-flash (8 train examples with previous actions) 58.0%;
  the 149M model alone 63.3% (+5.3 [+1.8, +8.9]); cascade 67.0%, **+9.0 [+6.3, +11.8] over few-shot Gemini**, with Gemini called on 38% of steps.
  Showing Gemini the dataset's conventions in context barely helps: few-shot − zero-shot = +0.9 [−1.0, +2.8].

## E16 (registered before running): cascade + app grounding + no-effect guard, live, fresh parameterisation
Motivation (disclosed): the E15 interim failure analysis (seed 30). The pointer put goal text into open_app ("note_SiFbv.txt in Markor",
"Android") with conf ≥ 0.96, and repeated confident actions that left the screen unchanged. Two general mechanisms (tools/aw_cascade_agent.py,
flags E16_APPS / E16_GUARD); neither reads the task:
(1) app grounding: the predicted app name is resolved against the device's launcher vocabulary (AndroidWorld adb_utils aliases). A whole-word
alias contained in the prediction wins, else the closest alias by character ratio ≥ 0.75; otherwise the step is deferred to the LLM.
(2) no-effect guard: if the screen is identical (texts, descriptions, boxes) to the screen before the pointer's previous action, the pointer may
not act; the LLM takes the step.
Test: all 116 AndroidWorld tasks with task_random_seed=31 (new parameters, unseen during design), n_task_combinations=1. Arms: E16 cascade
(t = 0.7, gemini-3.5-flash) vs T3A (gemini-3.5-flash) on the same seed. LLM_TRIES=8 for both arms. Tasks that fail to initialise, or end in an
infrastructure error (emulator/adb/LLM), in either arm are excluded and reported; any nan task is rerun once in its arm before analysis.
H16: E16 success ≥ T3A success on the paired valid tasks (report the difference with a paired bootstrap CI over tasks and an exact McNemar
test), with Gemini calls per task lower than T3A's. Secondary: the share of steps the pointer executes; E15 vs T3A on seed 30 (when complete).

## E17 (registered before any data is extracted): learning from corrections and consequences (self-improving cascade)
Idea: every step Gemini takes in a live run is a correction the on-device model can learn from, and the task checker says whether the
trajectory worked. Practice data = all steps (whoever executed them) of SUCCESSFUL episodes (AndroidWorld checker = 1.0) from the seed-30 runs
(T3A, E15 cascade, split), converted to the pointer's training format (click/long_press target = the chosen element's box centre).
Family split (fixed now): a task family is in PRACTICE set A iff int(md5(family_name).hexdigest(), 16) % 2 == 0, else HELD-OUT set B. Only
A-family episodes are used for training. No seed-31 data is ever used for training.
Model: the 149M goal-only pointer, fine-tuned from its E2 checkpoint on practice rows (A) mixed 1:4 with AndroidControl train rows (to limit
forgetting); recipe fixed before training: lr 2e-5, 1 epoch over the mix, same loss.
Test: E17 cascade (t = 0.7, E16 mechanisms on) on all 116 tasks, seed 31. Comparisons, paired on tasks valid in both arms:
H17a: E17 > E16 on A families (practice helps on new instances of practised task types).
H17b: E17 ≥ E16 on B families (no harm, possible transfer to unpractised task types in the same apps).
H17c: share of steps executed by the pointer increases vs E16 (fewer Gemini calls).
Also reported: E17 vs T3A (seed 31), overall and per family set. Offline check before the live run: AndroidControl test accuracy must not drop
by more than 2 points vs the E2 pointer (forgetting guard); if it does, the run is reported and not used.
- **Results (2026-10-03):** H16 NOT supported: E16 50.4% vs T3A 58.3% on 115 paired seed-31 tasks, −7.8 [−16.5, +0.9], McNemar p = 0.11
  (Gemini executed 70% of cascade steps). Secondary, E15 final (seed 30, 93 paired): 41.9% vs 54.8%, −12.9 [−21.5, −4.3], p = 0.008. The split
  agent (τ=0.9, 90 paired): 16.7% vs 56.7%. (Numbers corrected after fixing a task-name truncation that merged two tasks.) The offline per-step advantage (E14b) does not transfer to live multi-step tasks on unseen apps.
- **E17 amendment (before the data exists):** E17 (trained on seed-30 practice only, as registered; forgetting guard passed: AndroidControl
  test 61.0% vs E2 61.7%) is evaluated as registered. **E17b:** additional practice from T3A runs on seeds 40 and 41, A families only
  (bench/aw_results/families_A.txt); same extraction, filter, mix ratio (1:4) and recipe, trained from E2; evaluated on seed 31 like E17.
  Same hypotheses H17a–c for E17b.

## E18 (registered before any code or analysis): compile the frontier model into on-device programs (skill replay)
Thesis: a tiny local system does not need to out-reason Gemini; it can compile Gemini's verified successes into parameterised programs and
replay them exactly, with zero LLM calls, falling back to the LLM only when a program does not apply.
Mining (offline, automatic, no hand-written task rules): from successful T3A episodes (checker = 1.0) of PRACTICE families (set A, as E17)
on seeds 30, 40, 41 (never 31), build per family a goal template (literal text shared across instances; variable spans become slots; with a
single instance, slots are the goal substrings that reappear in typed text or clicked element text) and a step program (action type; target
described by text/desc/resource-id/class, with slot references; typed text with slot references; app names; waits).
Test-time agent (E18): sees only the goal. If some template's regex matches the goal, fill the slots and replay the program; each target is
located on the current screen by exact text/desc/resource-id (+class) match; if a target cannot be found, or no template matches, control
passes to T3A (gemini-3.5-flash) for the rest of the episode, given the executed steps as history. Replay steps count as zero LLM calls.
Test: all 116 tasks, seed 31, live, LLM_TRIES=8, nan rerun once. H18a: E18 > T3A on A families (paired, McNemar; CI over tasks).
H18b: E18 ≥ T3A overall. H18c: E18 uses fewer LLM calls per task than T3A. Also reported: template coverage (share of seed-31 A goals that match)
and replay success when a template matched.
- **E14c result (n = 999 of 1,000 registered steps; one step failed every retry and is excluded; 123 earlier failed calls rerun):** H14c
  SUPPORTED. gemini-3.1-pro-preview alone 60.6% (vs Flash 57.2%, +3.4 [+0.9, +6.0]); 149M alone 63.4% (+2.8 [−0.6, +6.4] vs Pro, n.s.);
  cascade 149M + Pro 69.3% (+8.7 [+6.2, +11.4] vs Pro alone, Pro on 38% of steps). Secondary: cascade 149M + Flash vs Pro alone +6.7 [+3.7, +10.0].

## E20 (registered before training): distil the frontier agent into an on-device 1.7B policy (zero frontier calls at test time)
Data: exact T3A prompts and gemini-3.5-flash outputs (action selection and step summary) from SUCCESSFUL episodes (checker = 1.0) of practice
families A only, from seeds 30, 31 (and 40, 41 when complete; E20 round 1 uses whatever exists at training time, reported). No B family and no
seed >= 32 data is ever used for training.
Model: Qwen/Qwen3-1.7B (non-thinking chat template), LoRA (r 64, all linear layers), loss on the output only, lr 1e-4, 2 epochs, max 8,192
tokens, merged for serving (vLLM on Modal).
Test: T3A agent with the distilled 1.7B as its only LLM (action + summary), live on all 116 tasks, seed 32; baseline T3A + gemini-3.5-flash on
seed 32; also zero-shot Qwen3-1.7B in T3A (no distillation) as a floor. Report A families (new parameters of practised task types) and
B families (held-out task types) separately; paired McNemar and task-bootstrap CIs; nan rerun once per arm.
H20a: distilled 1.7B > zero-shot 1.7B on B families. H20b (aspirational): distilled 1.7B ≥ T3A+Flash overall. Kill criterion for this line:
distilled 1.7B below 30% on B families.

## E21 (registered before any self-play data exists): self-play with verifiable rewards (expert iteration), A families only
Round k: the round-(k−1) distilled 1.7B (E20 = round 0) attempts every A family on two fresh practice seeds (50+2k, 51+2k) with sampling
temperature 0.7 inside T3A; AndroidWorld's checkers grade each episode; successful episodes' (prompt, output) pairs are added to the training
set; a new model is trained from the base with the E20 recipe on all accumulated pairs. Checkers are used only during practice; never at test.
Test after each round: live, seed 32 (same as E20), all 116 tasks, temperature 0, reporting A and B families; plus the offline B-family
agreement (bench/e20_offline.py). H21: B-family live success increases from round 0 to round 2 (paired, task-bootstrap CI). Stop rule: two
rounds without B-family improvement.
- **E20 result (round 0, live seed 32, final 116/116):** distilled Qwen3-1.7B, zero frontier calls: A families
  17/59 (29%), **B families 6/57 (11%)** → below the registered 30% kill threshold: H20b not supported; the distillation line as specified fails.
  Offline held-out next-action agreement (57–65%) did not translate into episode success: at ~60% per step, multi-step tasks compound to
  near zero. Floor (base 1.7B): 1/59 at last check.

## E22a (registered before any E22a run): does a local table memory + exact executor fix frontier agents' data questions?
Arms (both gemini-3.5-flash in T3A, both with compacted prompts GEMINI_COMPRESS=1, LLM_TRIES=8): (1) T3A; (2) T3A + table memory + exact
program executor (tools/aw_table_agent.py: rows harvested from every screen by tools/screen_table.py; "PROGRAM:" answers evaluated by
tools/screenql_exec.py with general parsers only). Tasks: the 25 data-question families (goal regex fixed in tools/aw_qa_dump.py), seeds 33
and 34 (fresh), paired. Development before the test used only A-family question episodes (seeds 30/31 screens, seed 40 smoke runs).
H22a: arm 2 > arm 1 on question-task success (paired over 50 instances, McNemar; task-bootstrap CI). Reported per A/B family set, plus the
share of answers given as programs. If H22a holds, E22b replaces Gemini by the distilled 1.7B in arm 2.
- **E22a postponed (2026-10-04):** Vertex returns RESOURCE_EXHAUSTED on nearly every call even for a single agent (project quota), so the
  Gemini arm cannot run.

## E22b (registered before any E22b run): the on-device 1.7B + table memory + exact executor on data questions, zero frontier calls
Agent: T3A whose only LLM is the round-1 distilled Qwen3-1.7B (serve-r1, guided decoding), plus tools/aw_table_agent.py. Test: the B-family
question tasks (families never trained on and never inspected during E22 development), seeds 30 and 31, compared paired with the existing
T3A + gemini-3.5-flash results on the same task instances (full_t3a, t3a_s31). A-family question tasks on the same seeds are reported as
exploratory only (their screens were used for development). H22b: success on B question tasks ≥ Gemini's on the same instances.
- **E22b result (2026-10-04):** on the 22 B-family question instances (seeds 30/31), the on-device system (round-1 1.7B policy, unguided,
  + table memory + 1.7B translator 94% on held-out synthetic + exact executor; zero frontier calls) solved **3/22 vs T3A+gemini-3.5-flash 8/22**
  on the same instances (ours-only 1, Gemini-only 6). **H22b NOT supported.** Failure anatomy: most tasks never reached an answer (navigation
  within the 10-step budget; 0–3 rows harvested), and on real screens the translator mis-read column semantics (clock time vs duration,
  absent columns) learned from synthetic names. The exact executor was correct whenever the needed rows were on screen (TasksDueOnDate 2/2).
  The exploratory A-family run was stopped to save cost.

## E23 (registered before any E23 code): compile the app by self-exploration, answer exactly (no step-by-step neural navigation)
Thesis: compounding per-step errors sink small models; make navigation exact. (1) Task-free exploration on practice instances (seeds ≥ 50,
any family's initial app state) builds per-app screen graphs with STRUCTURAL fingerprints (resource ids + classes, no texts) so graphs merge
across data; screens whose rows segment into tables (tools/screen_table.py) are "data views". (2) Test agent for question goals: open the app
named in the goal (launcher-vocabulary resolution), BFS the shortest verified path to the data view whose column names/sample values best
match the question words, replay it by element identity, scroll to harvest all rows, then ONE translator call (≤1.7B, typed schema inferred
generically from values: clock / duration / distance / date / number / text) and the exact executor. No frontier model, no per-step policy.
Test: all 25 question families, fresh seeds 32 and 33 (50 instances), live AndroidWorld with the official step budgets.
Comparator (no GCP allowed): T3A + gemini-3.5-flash's success rate on the same 25 families from seeds 30/31 (19/50 = 38%); unpaired,
disclosed. H23: E23 success on the 50 instances > 38% (one-sided binomial test vs 0.38). Secondary: per family set A/B, and
the share of instances where the needed rows were harvested.

## W3 (registered 2026-10-05, after test 3 was committed at 625a2a3 and before any model ran on it): 0.6B translator vs 14B/32B on fresh pages
Test 3: 8 new pages captured logged-out (Amazon x3, eBay, Walmart, GitHub, Wikipedia, Hacker News), 50 requests written before any model run.
Comparators: Qwen3-14B-AWQ and Qwen3-32B-AWQ reading the page (same cards + shortlist + instructions as tests 2/2b; Kaggle 2xT4).
Primary translator, chosen on DEV data only (tests 1+2+2b, 135 cases): v6 (gen_screenql v6 data, v3 recipe) if its dev total >= v3's
(100/135), else v3. Scored with bench/act_eval_sql.py (SQL_V2=1). H-W3: the 0.6B translator scores > the better of 14B/32B on test 3
(exact McNemar on the paired 50 cases reported; a tie is reported as a tie).
- **W3, our side (recorded before the 14B/32B results exist):** dev totals (tests 1+2+2b): v6 104/135 (38, 29, 37) vs v3 100/135 → v6 is
  primary per the rule. **Test 3: v6 35/50 (70%)**, v3 34/50. Per-case outputs: bench/results/w3_v6_t3.txt, w3_v3_t3.txt. First Kaggle
  baseline run failed on an environment error (torchaudio/CUDA mismatch after the vLLM upgrade); fixed and re-run, no change to prompts.
- **W3 result: H-W3 NOT supported.** Test 3 (50): 0.6B v6 35/50; Qwen3-14B-AWQ 31/50 (ours-only 11, theirs-only 7, McNemar p = 0.48);
  Qwen3-32B-AWQ 42/50 (ours-only 5, theirs-only 12, p = 0.14). The 0.6B translator beats the 14B and trails the 32B on fresh pages; neither
  difference is significant at n = 50. Baseline rows: bench/results/w3_baselines_t3.json.

## W4 (registered 2026-10-05, before test 4 exists): frozen 0.6B system vs 14B/32B on fresh pages, requests written blind
Why: after W3 the executor was improved on failures from tests 1–3 (v7-lite → v9, all generic rules; see docs/LOG.md), so test 3 is dev.
System frozen NOW, before any test-4 request is written: translator v6 (training/adapters/v6_06, Qwen3-0.6B-4bit), executor tools/screenql.py
at commit 0a32e9b, scored by bench/act_eval_sql.py with SQL_V2=1 (temp 0, K=1). No changes to translator, executor or prompts after this line
until W4 is reported.
Test 4: 9 new pages captured logged-out in Incognito while the user was idle (Amazon kettle, eBay film camera, Walmart cordless vacuum, Newegg
NVMe SSD, GitHub django, Wikipedia Volcano, Stack Overflow "yield", MDN Fetch API, Hacker News newest). Requests (~54) and gold ids written by a
separate agent that may read ONLY the 9 page files (not the executor, translator, training data or prior tests); bench/act_eval4.json is committed
before any model runs on it. Comparators: Qwen3-14B-AWQ and Qwen3-32B-AWQ with the identical W3 page-reading setup (Kaggle 2xT4, same prompts).
H-W4 (primary): 0.6B system > Qwen3-32B-AWQ on test 4, exact McNemar on paired cases; "beats" claimed only if ours > theirs AND p < 0.05,
otherwise reported as ahead/tie/behind with p. Secondary: vs 14B (same test); test-4 score of the Milo app with the sidecar (routing frozen at b6876de).
- **W4 result: H-W4 NOT supported.** Test 4 (54, blind-written): frozen 0.6B system 41/54 (76%); Qwen3-14B-AWQ 45/54 (ours-only 5, theirs-only 9,
  McNemar p = 0.42); Qwen3-32B-AWQ 45/54 (ours-only 6, theirs-only 10, p = 0.45). The 0.6B system trails both by 4 cases; not significant.
  Secondary: Milo app (routing b6876de, v6 sidecar) 32/54 vs the app without the sidecar 20/54 (with-only 13, without-only 1, p = 0.002).
  Files: bench/results/w4_v6_t4.txt, w4_baselines_t4.json, w4_milo_app_t4.txt, w4_milo_app_noscreenql_t4.txt.

## W5 (registered 2026-10-05, before test 5 exists): frozen v7 system vs 14B/32B, blind-written fresh test
After W4, executor v10–v12 and translator v7 were developed on tests 1–4 (all dev). Frozen NOW: translator training/adapters/v7_06
(Qwen3-0.6B-4bit), executor tools/screenql.py as of commit 6e4d2c2, bench/act_eval_sql.py SQL_V2=1, temp 0, K=1. No changes until W5 is reported.
Test 5: ≥8 new logged-out pages from sites/queries not used in tests 1–4 (captured in Incognito only while the user is idle), ~54 requests
written by a separate agent that may read only those page files, same act mix and gold rules as test 4; committed before any model runs.
Comparators: Qwen3-14B-AWQ, Qwen3-32B-AWQ, same Kaggle setup and prompts as W3/W4.
H-W5: 0.6B system > Qwen3-32B-AWQ (exact McNemar; "beats" only if ahead AND p < 0.05; otherwise reported as ahead/tie/behind with p).
- **W5 result: H-W5 NOT supported.** Test 5 (60, blind-written, 10 pages): frozen 0.6B v7 system 47/60 (78%); Qwen3-14B-AWQ 46/60
  (ours-only 10, theirs-only 9, p = 1.00); Qwen3-32B-AWQ 52/60 (ours-only 4, theirs-only 9, p = 0.27). Ties 14B, trails 32B by 5 (n.s.).
  Secondary: Milo app with sidecar 35/60 vs 21/60 without (with-only 14, without-only 0, p = 0.0001).

## W6 (registered 2026-10-05, before test 6 exists): does executor work since W5 generalise?
Since W5: executor v13–v15 (fixes from test-5 failures, teacher-free self-practice on practice pages, Qwen3-32B-written practice requests);
translator v8 (teacher distillation) was worse on dev and is NOT used. Frozen NOW: translator v7 (training/adapters/v7_06), executor
tools/screenql.py at this commit, bench/act_eval_sql.py SQL_V2=1, temp 0. Test 6: ≥8 fresh logged-out pages (sites/queries unused in tests 1–5 and
practice pages), ~60 requests written by a separate agent that reads only those page files; committed before any model runs.
Comparators: Qwen3-14B-AWQ / Qwen3-32B-AWQ, same Kaggle setup. H-W6: 0.6B > Qwen3-32B-AWQ (exact McNemar; "beats" only if ahead AND p < 0.05).
Secondary: frozen v12-era system (v7 + executor at 6e4d2c2) on test 6, to measure what v13–v15 added on fresh pages.
- **W6 result: H-W6 NOT supported (ahead, not significant).** Test 6 (60, blind, 12 pages): frozen 0.6B v7 + executor v15 53/60 (88%);
  Qwen3-32B-AWQ 50/60 (ours-only 7, theirs-only 4, p = 0.55); Qwen3-14B-AWQ 45/60 (ours-only 10, theirs-only 2, p = 0.039 — a listed comparator,
  not the primary hypothesis; reported as such). First fresh test where the 0.6B system scores above 32B. Secondary: executor as of W5 (6e4d2c2)
  52/60 → the v13–v15 work added +1 on fresh pages.

## W7 (registered 2026-10-05, before test 7 exists): a powered test
Why: W6 had the 0.6B system ahead of Qwen3-32B by 3/60 — far too few discordant pairs to decide. W7 is ~4× larger.
Frozen NOW: translator v7 (training/adapters/v7_06), executor tools/screenql.py at this commit (v16), bench/act_eval_sql.py SQL_V2=1, temp 0.
Test 7: ~40 new logged-out pages (sites/queries unused in tests 1–6 and in practice pages; captured only while the user is idle), ~6 requests per
page (~240) written by separate agents that read only those page files; same act mix/gold rules as tests 4–6; committed before any model runs.
Comparators: Qwen3-14B-AWQ and Qwen3-32B-AWQ, same Kaggle setup and prompts.
H-W7 (primary): 0.6B system > Qwen3-32B-AWQ, exact McNemar, "beats" only if ahead AND p < 0.05; otherwise ahead/tie/behind with p.
Secondary: vs 14B; the Milo app with/without the sidecar (routing at this commit).
- **W7 result: H-W7 NOT supported — a tie with 32B.** Test 7 (258 requests, 43 pages, written blind by 3 agents): frozen 0.6B (v7 + executor v16)
  220/258 (85.3%); Qwen3-32B-AWQ 217/258 (84.1%; ours-only 32, theirs-only 29, p = 0.80; difference +1.2 pts, approx. 95% CI −4.7 to +7.1);
  Qwen3-14B-AWQ 199/258 (77.1%; ours-only 46, theirs-only 25, p = 0.017, secondary). Milo app: 190/258 with sidecar vs 115/258 without
  (with-only 76, without-only 1, p ≈ 1e-21). Files: bench/results/w7_*.

## W8 (registered 2026-10-06, before test 8 requests exist): replication with the improved system + non-inferiority
Since W7 (all chosen on dev tests 1–7): executor v17; a confidence-gated 0.6B chooser (training/adapters/chooser9_06, trained only on practice
pages: Qwen3-32B-verified practice requests + self-practice) that may override a weak executor match only when its probability ≥ 0.97
(tools/choose.py). Dev replay tests 1–7: 548 → 553. Frozen NOW: translator v7 (training/adapters/v7_06), executor tools/screenql.py at this
commit, chooser9_06 with τ = 0.97, scored by bench/act_eval_sql.py with SQL_V2=1 SQL_CHOOSER=training/adapters/chooser9_06 SQL_TAU=0.97.
Test 8: 44 new logged-out pages captured while the user was idle (kept outside training/pages until this registration), ~6 requests/page written
by separate agents that read only those page files; committed before any model runs. Comparators: Qwen3-14B-AWQ, Qwen3-32B-AWQ (same setup).
H-W8a (primary, superiority): 0.6B > Qwen3-32B-AWQ, exact McNemar p < 0.05.
H-W8b (non-inferiority): the lower bound of the 95% CI of (0.6B − 32B) accuracy, paired (Wald on discordant pairs), is above −5 points.
Also reported: pooled W7+W8 with the same two tests; vs 14B; the Milo app with the sidecar (translator + executor + chooser) vs without.
- W8 test written (before any model run): 264 requests on 44 pages by 3 agents. Disclosure: one writer's dump script globbed all t8 page files
  (and some t7) into its scratch space; it reports using none of their contents and deleted the copies. No writer read any system code.
  Some page names differ from content (e.g. t8_wiki_moon is Neptune, t8_mdn_fetch_api is an MDN 404 page); requests follow the content.
- **W8 result.** Test 8 (264 blind requests, 44 pages): frozen 0.6B system 220/264 (83.3%); Qwen3-32B-AWQ 215/264 (81.4%); Qwen3-14B-AWQ 196/264 (74.2%).
  H-W8a (superiority over 32B): NOT supported — +1.9 pts, 95% CI [−4.4, +8.1], McNemar p = 0.64 (ours-only 38, theirs-only 33).
  **H-W8b (non-inferiority, margin 5 pts): SUPPORTED** — lower CI bound −4.4 > −5.
  vs 14B: +9.1 pts [+2.5, +15.6], p = 0.010. Without the chooser our system scores 215/264 (chooser +5 on fresh pages).
  Pooled W7+W8 (522): vs 32B 440 vs 432, +1.5 [−2.8, +5.8], p = 0.54; vs 14B 440 vs 395, +8.6 [+4.1, +13.2], p < 0.001.
  Milo app on W8: 191/264 with the sidecar vs 115/264 without (with-only 78, without-only 2).

## W9 (registered 2026-10-07, before test 9 requests exist)
Since W8 (chosen on dev tests 1–8): executor v18 (generic rules from W8 failures). Negative and NOT used: page-aware translator (662 vs 773 on dev).
Frozen NOW: translator v7 (training/adapters/v7_06), executor tools/screenql.py at this commit (v18), chooser training/adapters/chooser9_06 with
τ = 0.90 (best on dev: tests 1–8 = 786/881 vs 773 without chooser), scored by bench/act_eval_sql.py SQL_V2=1 SQL_CHOOSER=training/adapters/chooser9_06
SQL_TAU=0.90. Test 9: 43 new logged-out pages (captured while the user was idle, kept outside training/pages until now), ~6 requests per page by
3 separate agents that read only their own page files (instructed not to glob other files); committed before any model runs.
Comparators: Qwen3-14B-AWQ, Qwen3-32B-AWQ (same Kaggle setup and prompts).
H-W9a (primary): 0.6B > Qwen3-32B on test 9 (exact McNemar, p < 0.05).
H-W9b: non-inferiority on test 9, lower 95% CI bound of (0.6B − 32B) > −5 points.
H-W9c (pooled, pre-registered now): over W7+W8+W9 (each with the system frozen for that test), 0.6B > Qwen3-32B, exact McNemar on all pooled
paired requests, p < 0.05.
- **W9 result.** Test 9 (258 blind requests, 43 pages): frozen 0.6B system 225/258 (87.2%); Qwen3-32B-AWQ 208/258 (80.6%); Qwen3-14B-AWQ 193/258 (74.8%).
  **H-W9a (superiority over 32B): SUPPORTED** — +6.6 pts, 95% CI [+0.8, +12.4], exact McNemar p = 0.036 (ours-only 38, theirs-only 21).
  **H-W9b (non-inferiority, margin 5): SUPPORTED.** vs 14B: +12.4 [+6.1, +18.7], p = 0.0002.
  **H-W9c (pooled W7+W8+W9 superiority over 32B): NOT supported** — 665/780 vs 640/780, +3.2 [−0.3, +6.7], p = 0.082 (ours-only 108, theirs-only 83).
  Pooled vs 14B: 665 vs 588, +9.9 [+6.2, +13.6], p < 0.0001.
  Milo app on W9: 199/258 with the sidecar vs 105/258 without (with-only 95, without-only 1).
  Caveat (stated with the result): superiority over 32B was also registered for W3–W8 and not supported; W9 is the first single test to pass,
  and the registered pooled test does not. The system improved between rounds (W7 → W9: 85.3% → 87.2%) while 32B ranged 80.6–84.1%.

## W10 (registered 2026-10-07, right after W9, before test 10 exists): confirmatory replication, NO system changes
Frozen system identical to W9 (translator v7, executor at commit 54835d2, chooser9_06 τ = 0.90; same scorer). Test 10: ~43 new logged-out pages
(sites/queries unused in all earlier tests and practice pages), ~258 requests by 3 separate agents reading only their own pages by exact path.
H-W10a: 0.6B > Qwen3-32B-AWQ on test 10 (exact McNemar, p < 0.05).
H-W10b (pooled confirmatory, same frozen system): over W9+W10, 0.6B > Qwen3-32B-AWQ (exact McNemar, p < 0.05).
H-W10c: non-inferiority on test 10 (lower 95% CI bound > −5 points).
- **W10 result (confirmatory, identical frozen system).** Test 10 (264 blind requests, 44 pages; isolated crawler): 0.6B 214/264 (81.1%);
  Qwen3-32B-AWQ 210/264 (79.5%); Qwen3-14B-AWQ 204/264 (77.3%).
  **H-W10a (superiority on W10): NOT supported** — +1.5 pts [−4.9, +7.9], p = 0.73 (ours-only 39, theirs-only 35).
  **H-W10b (pooled W9+W10 superiority): NOT supported** — 439 vs 418 of 522, +4.0 [−0.3, +8.3], p = 0.083.
  **H-W10c (non-inferiority, margin 5): SUPPORTED** (lower bound −4.9).
  The W9 superiority result did not replicate. Over all four large fresh tests W7–W10 (1,044 requests): 0.6B 879, Qwen3-32B 850
  (+2.8 [−0.3, +5.8], p = 0.085), Qwen3-14B 792 (+8.3 [+5.1, +11.5], p < 0.0001).

## W11 (registered 2026-10-24, before test 11 exists): local-first cascade vs a frontier model
Context (exploratory, on W7–W10, now development data): with a fair prompt the 0.6B system ties Qwen3-14B/32B and Claude Haiku 4.5 and
trails Claude Sonnet 5.5 (879 vs 951 of 1,044). A cascade simulated on that data matched or beat Sonnet at a fraction of its cost.
Frozen system: translator v7, executor + chooser9_06 τ = 0.90 as in bench/cascade_sim.py at commit 3abd4f1 (no sampling), router
bench/router_frozen.json (logistic regression on the local run's own signals, fit on W7–W10; request kept local when score ≥ cut_local50).
Cloud models: claude-haiku-4-5-20251001 (gen_act.SYSTEM_V2, shortlist k=40) and claude-sonnet-5-5 (same, adaptive thinking, effort low),
Batch API, prompts unchanged. Three-tier cascade: router-local → else Haiku; if Haiku's action equals ours, ours; else Sonnet's.
Test 11: ~86 new logged-out pages from sites/queries unused in all earlier tests and practice pages (isolated crawler, user idle), ~6
requests per page by separate agents that read only their own page files; committed before any model runs.
H-W11a (primary): three-tier cascade non-inferior to Sonnet alone, lower 95% CI bound of (cascade − Sonnet) > −3 points.
H-W11b: the three-tier cascade keeps ≥ 40% of requests on device and costs ≤ 40% of Sonnet alone (Batch API token costs as billed).
H-W11c: router-only cascade (router-local, else Sonnet) non-inferior to Sonnet alone, margin 3 points.
H-W11d: 0.6B alone non-inferior to Qwen3-32B-AWQ with SYSTEM_V2 (margin 5 points).
Reported regardless of outcome: every number above, the local share, accuracy of the local subset, cost and latency per request.
- Test 11 committed before any model run: 444 requests over 74 pages (5 writer agents, 15/15/15/15/14 pages, page files only).
  Deviations, stated now: (1) the captured page files initially lacked the explicit 1-based "id" field; writers were told mid-task, all
  reloaded and re-checked their ids (lexical sanity check: no gold set is better explained by an off-by-one). (2) Two writers shared a
  scratchpad; writer 0 ran writer 1's build script once (regenerating part1 from writer 1's final script; counts match writer 1's report)
  and saw one request line and the page names of group 1. No gold was changed.
- **W11 result (444 blind requests, 74 pages; frozen system, frozen router; scored by bench/w11_score.py written before the test).**
  Router kept 51.8% of requests on device (those were 96.1% correct); Sonnet called on 18.0%.
  **H-W11a (three-tier cascade non-inferior to Sonnet 5.5, margin 3): SUPPORTED** — 421 vs 419, +0.5 [−1.4, +2.3], p = 0.82 (+10/−8).
  **H-W11b (≥40% on device, ≤40% of Sonnet's cost): SUPPORTED** — 51.8% on device; $0.62 vs $2.10 per 1k requests (29.6%).
  **H-W11c (router-only cascade non-inferior to Sonnet, margin 3): SUPPORTED** — 419 vs 419, +0.0 [−1.5, +1.5].
  **H-W11d (0.6B non-inferior to Qwen3-32B-AWQ with SYSTEM_V2, margin 5): SUPPORTED** — 394 vs 384, +2.3 [−1.6, +6.1], p = 0.31.
  Qwen3-14B-AWQ (v2) 391 (p = 0.80).
  Context (not registered): 0.6B alone 394 (88.7%), Haiku 4.5 379 (85.4%; +3.4 [−0.1, +6.8], p=0.072), Sonnet 5.5 419 (94.4%).

## W12 (registered 2026-10-24, after W11, before test 12 exists): replication of the W11 cascade, NO changes
Identical frozen system, router (bench/router_frozen.json, not refit on W11), cloud models, prompts and scorer (bench/w11_score.py run on
test 12). Test 12: ~75 new logged-out pages from sites/queries unused in all earlier tests and practice pages, ~6 requests per page by
separate writer agents (page files only, unique scratch file names). Committed before any model runs.
H-W12a: three-tier cascade non-inferior to Sonnet 5.5 alone (margin 3 points).  H-W12b: ≥40% on device and ≤40% of Sonnet's cost.
H-W12c (pooled W11+W12, primary for the paper): three-tier cascade non-inferior to Sonnet, margin 2 points.
- Test 12 committed before any model run: 480 requests over 80 pages (5 writer agents x 16 pages, page files only, ids present from the
  start, unique helper names; no integrity deviations reported).
- **W12 result (replication; identical frozen system and router; 480 blind requests, 80 pages).** Router kept 45.6% on device (95.0%
  correct there); Sonnet called on 21.0%.
  **H-W12a (cascade non-inferior to Sonnet, margin 3): SUPPORTED** — 453 vs 444, +1.9 [−0.3, +4.1], p = 0.14 (+19/−10).
  **H-W12b (≥40% on device, ≤40% of cost): SUPPORTED** — 45.6% on device; $0.69 vs $2.03 per 1k (34.1%).
  **H-W12c (pooled W11+W12, margin 2; primary): SUPPORTED** — 874 vs 863 of 924, +1.2 [−0.3, +2.6], p = 0.14 (+29/−18);
  48.6% on device, 31.9% of Sonnet's cost.
  Context: 0.6B alone 418, Haiku 4.5 408, Sonnet 5.5 444 of 480. Router-only cascade 445 vs 444.

## AC (registered 2026-10-24, before any frontier model sees the sampled AndroidControl test steps): the cascade on a public, human-labelled benchmark
Steps: 1,000 AndroidControl test steps drawn uniformly with seed 2027 (bench/ac_sample1000.json; all 8,444 test steps are the population).
Labels: AndroidControl's human demonstrations; scored with the paper's strict metric (bench/pointer_eval_local.py rule, re-implemented in
bench/ac_frontier.py). Local model: the frozen 149M pointer (training/ac/pointer_v2; its per-step results were computed earlier and are not
changed). Cloud: Claude Haiku 4.5 and Sonnet 5.5 (adaptive thinking, effort low), Batch API, prompt bench/ac_frontier.py SYSTEM (fixed after a
20-step pilot on steps outside the sample: reply limit raised, the dataset's scroll naming stated). Inputs match the pointer's: instruction,
history/consequence string, goal, first 220 elements in the pointer's rendering. Router and cascade rule: bench/ac_cascade_score.py
(confidence from predicted, not gold, type; cut = median over all test steps, label-free).
H-ACa (primary): three-tier cascade non-inferior to Sonnet alone, margin 3 points.
H-ACb: ≥40% of steps on device and ≤40% of Sonnet's cost.
H-ACc (secondary, two-sided, no direction registered): 149M pointer alone vs Sonnet alone.
- **AC result (1,000 sampled AndroidControl test steps; human labels; strict metric).** Router kept 49.8% on device (95.6% correct there);
  Sonnet called on 27.2%.
  **H-ACa (cascade non-inferior to Sonnet, margin 3): SUPPORTED** — 842 vs 804, +3.8 [+2.2, +5.4], p < 1e-4 (+53/−15).
  **H-ACb (≥40% on device AND ≤40% of Sonnet's cost): NOT supported** — 49.8% on device, but cost 42.4% of Sonnet ($1.91 vs $4.50 per 1k).
  **H-ACc (149M pointer alone vs Sonnet, two-sided): pointer better** — 878 vs 804, +7.4 [+4.9, +9.9], p < 1e-5 (+123/−49).
  Context: Haiku 4.5 605. The cascade (842) is below the pointer alone (878): here the frontier fallback hurts.
  Exploratory breakdown: pointer > Sonnet on every official subsplit (IDD 90.4 vs 80.6; app_unseen 87.3 vs 81.8, p=0.005; task_unseen
  85.2 vs 79.8, p=0.004; category_unseen 84.8 vs 79.2, p=0.007). Much of the margin is dataset convention learned in training (wait 89.7
  vs 42.6, navigate_back 95.3 vs 65.1); clicks 85.4 vs 83.6. Excluding wait/long_press: 815 vs 773 of 930 (+4.5 pts, p=0.0004).
  Caveat: the pointer is fine-tuned on AndroidControl train; Claude is zero-shot.

## AC-Opus (registered 2026-10-24, before Opus sees the sample): the strongest available frontier model on the same 1,000 steps
Same 1,000 AndroidControl steps, prompt, inputs and strict scorer as AC; model claude-opus-5-5 (adaptive thinking, effort low; Batch API).
H-ACO (two-sided, no direction registered): 149M pointer alone vs Opus 5.5 alone, exact McNemar on paired steps.
Reported regardless: Opus accuracy overall and per official subsplit, cost per 1k steps.
- **AC-Opus result.** Opus 5.5 800/1,000 ($7.41 per 1k steps). **H-ACO: pointer better** — 878 vs 800, +7.8 [+5.2, +10.4], p < 1e-6
  (+133/−55). Per subsplit: IDD 90.4 vs 82.7; app_unseen 87.3 vs 79.8 (p=0.0008); task_unseen 85.2 vs 77.1 (p=0.0001); category_unseen
  84.8 vs 76.6 (p=0.0004). Excluding wait/long_press 815 vs 761 of 930 (p=2e-5). Taps alone 85.4 vs 84.4 (p=0.59, tie).

## E24 (registered 2026-10-25, before any E24 episode): self-improvement from environment reward, zero frontier calls
Policy: Qwen3-1.7B. Round 0 = E20 round-1 policy (~/mlx_models/r1_q4, 4-bit MLX, served on the Mac), the same agent wrapper (t3a_milo).
Split: AndroidWorld task families A (practice) / B (held-out) by the pre-registered md5 rule (tools/aw_practice_extract.py). Nothing
from B is ever trained on. Each round: the current policy runs every A family on fresh practice seeds (50, 51, then new seeds per round)
at temperature 0.7; only episodes the AndroidWorld checker scores 1.0 become training pairs (no frontier model, no human labels);
the next policy = Qwen3-1.7B LoRA on (E20 pairs + all successful self-generated pairs so far), same recipe as E20 round 1.
Evaluation each round: all 116 tasks on a fresh evaluation seed never used for practice (first: 60), greedy decoding, success by the
benchmark checkers; paired against round 0 on the same seed.
H-E24a (primary): after round 2, B-family success > round 0 on the same evaluation seed (exact McNemar on paired B tasks, p < 0.05).
H-E24b: after round 2, A-family success > round 0 (same test).
Kill criterion: stop if round 1 improves B by fewer than 3 tasks over round 0 on the same seed.
Budget: AWS ≤ $60 total (emulator hosts only, stopped when idle); the policy runs on the Mac; fine-tuning on free Kaggle GPUs.
- **E24c (registered 2026-10-25, before any guard episode):** hard no-progress guard (tools/aw_guard.py: refuse an action already taken
  with no effect on the same screen; resample at temp 0.8 up to 4×; else navigate_back). Offline replay on E22b 1.7B episodes: would block
  24/143 steps. Arm t3a_milo_guard with the round-0 policy on evaluation seed 60 (all 116 tasks, greedy), paired with the round-0 plain arm.
  H-E24c: guard > plain on all 116 tasks (exact McNemar, p < 0.05). Later rounds use the guard if H-E24c holds.
- **E24 amendment (2026-10-25, before any round-1 training data is built):** round data come from tools/aw_round_data.py — successful A-family
  episodes, with no-effect steps dropped (element list unchanged; AppVLM) and fork-point steps (screen shared with a failed episode of the
  same family, different action; UI-Voyager) weighted 3x. Motivation: 2026 literature refresh (docs/LOG.md). Hypotheses and kill criterion unchanged.

## E24 amendment 2 (2026-10-25 13:20 EDT, before any round-1 data is built or trained)
Round-0 practice is extended with seeds 52 and 53 (same A families, same policy r1_q4, temperature 0.7) on a second host (milo-aw3), so that
each A family has up to 4 rollouts and fork points can exist. Round-1 training data = aw_round_data.py over all valid round-0 practice runs
(s50, s51, s52, s53; the tunnel-dead s51 attempt is excluded). Evaluation, held-out B families, metric and kill criterion are unchanged.

## E24 amendment 3 (2026-10-25 14:50 EDT, before any round-1 data is built; no eval episode has run)
For speed (one Mac serves both hosts; no GPU quota on AWS): practice seed 53 is dropped, so round-1 data = s50, s51, s52 (up to 3 rollouts
per A family). The round-0 eval (seed 60, greedy) and the guard arm are split across the two hosts by alternating task name (58 + 58).
AndroidWorld seeds each task instance from (seed, task name), so the instances are the same as in a single-host run. Scores are merged.

## E24 notes (2026-10-26 22:10 EDT, before any round-0 eval or guard result is scored)
- Implementation bug fixed in aw_guard.py: its navigate_back fallback returned an empty raw response, which T3A treats as an LLM failure
  (episode -> nan, excluded). 4 guard episodes hit it; they are rerun with the fix. Design unchanged.
- A ~14 h Mac outage turned 76 of 116 round-0 eval episodes into nan (infra) episodes; they are moved aside and rerun (same instances).
- Round-1 data built as amended: 175 practice episodes (s50-52), 44 successes, 23 A families, 396 pairs, 5 fork points, 66 no-effect
  steps dropped. Trained on Kaggle T4 (kernel milo-aw-sft-r1) over sft_r1 + these pairs; 4-bit conversion with the same quantization
  config as r1_q4, done on the AWS host with MLX CPU (Mac disk is full).

## E24 amendment 4 (2026-10-27 18:45 EDT, before any round-1 eval episode has run)
- Scheduling only: round-2 practice (round-1 policy r1_round1_q4, A families only, temperature 0.7, fresh seeds 54, 55, 56, split across
  the two hosts by alternating task name) is queued on each host right after its half of the round-1 eval, because both use the same
  policy and the Mac can serve only one model. If the kill criterion fails on the round-1 eval, practice is stopped and its data unused.
- Scoring script fixed in advance: tools/e24_compare.py (paired on tasks valid in both rounds; exact two-sided McNemar; nan = infra, excluded).
  Kill check = (#B tasks solved by round 1 and not round 0) - (#B solved by round 0 and not round 1) >= 3 on paired B tasks.

## E24e (registered 2026-10-27 23:26 EDT, before any t3a_milo_done episode): refuse unsupported completion claims
Motivation: in round-0 practice (A families, s50-52) 67 of 111 completion claims were false. Arm t3a_milo_done (tools/aw_done_guard.py)
= the plain agent plus a task-agnostic check on each "complete" claim: refuse if a commit button (save/ok/done/apply/confirm/add/create/
send/submit) or a cancel/discard dialog is on screen, the previous action was navigate_back (questions exempt), or a goal literal (quoted
string, file name, phone number; none for questions or removal goals) is missing from the screen. A refused claim is resampled at
temperature 0.8 (up to 2 tries), at most 2 refusals per episode. Designed on A-family practice only (replay: refuses 15/67 false claims,
1/44 true ones); no B-family or seed-60 episode was inspected for its design.
Run: policy r1_round1_q4, evaluation seed 60, all 116 tasks, greedy, split across hosts as in amendment 3; paired with the round-1 plain
eval (e24_r1_eval_s60_*). Scheduled when hosts are free (immediately if the round-1 kill check fails; else after round-2 practice).
H-E24e: done-check > plain on all 116 tasks (exact two-sided McNemar on paired valid tasks, p < 0.05, tools/e24_compare.py).
Secondary (descriptive): false-completion rate, and success on B families.

## E24e amendment (2026-10-27 23:28 EDT; round-1 eval in progress, not scored; no t3a_milo_done episode has run)
- Scope: the done-check acts only when the agent claims completion, so on a task whose round-1 plain episode never claimed completion
  the arm is the plain agent. E24e therefore runs only the tasks whose round-1 plain (seed 60) episode ended with status complete; every
  other task is counted with its plain outcome for both arms (zero discordant pairs by construction). H-E24e and its test are unchanged
  (exact McNemar over discordant pairs, which can arise only in the run set).
- Order: on each host, E24e runs right after its half of the round-1 eval, then round-2 practice (amendment 4) follows.

## AC-Fair (registered 2026-10-28 12:16 EDT, before any few-shot frontier request): does the pointer still win when Claude sees the dataset's conventions?
Objection being tested: AC/AC-Opus compared a 149M model trained on AndroidControl TRAIN with zero-shot Claude. Opus's zero-shot errors
include 28 scroll_down→scroll_up and 27 wait→click, i.e. labelling conventions it could learn from examples.
Arms (same 1,000 steps, bench/ac_sample1000.json; same SYSTEM prompt, screen rendering and strict metric as bench/ac_frontier.py):
- Opus-FS: claude-opus-5-5, effort medium (its default), plus 16 retrieved TRAIN examples per step.
- Fable-FS: claude-fable-5-1, effort medium, same 16 examples.
Retrieval (bench/ac_fewshot.py, fixed before any result): TF-IDF cosine between the test step's "goal + instruction" and every TRAIN step's,
top 16 (ties by row order). Each example shows goal, instruction, last-step string, the gold action as JSON and, for click/long_press, the
target element rendered with el_text(). TRAIN only; no test step, test episode or eval answer is shown. The 149M pointer saw all of TRAIN in
training, so this gives Claude the same data, retrieved for the step at hand.
Hypotheses (paired exact McNemar vs the frozen pointer_v2 strict outcomes on the same steps, two-sided α = 0.05):
H-AF1: pointer > Opus-FS.  H-AF2: pointer > Fable-FS.  Also reported: few-shot − zero-shot for Opus; unseen-app subsplit; cost per 1k steps.
Kill rule: if either arm is ≥ the pointer (878), we stop claiming "the 149M model beats frontier models" and say so in the paper.
Prices used: Opus 5.5 $4/$20, Fable 5.1 $10/$50 per MTok, batch 50%. Spend cap for AC-Fair: $45.
- **AC-Fair result (2026-10-28): KILL RULE TRIGGERED.** With 16 retrieved TRAIN examples, Opus 5.5 scores 893 and Fable 5.1 870 of 1,000
  (pointer 878). H-AF1 NOT supported: pointer − Opus-FS = −1.5 (+45/−60, p = 0.17). H-AF2 NOT supported: +0.8 (+63/−55, p = 0.52).
  Unseen apps: pointer 350, Opus-FS 359, Fable-FS 350 of 401. Few-shot lifts Opus by +9.3 (800 → 893). Cost per 1k steps: Opus-FS $9.44,
  Fable-FS $23.36. We no longer claim the 149M model beats frontier models; the supported claim is a statistical tie with few-shot Opus 5.5
  and Fable 5.1. Exploratory (not registered): pointer on its 70% most confident steps + Opus-FS on the rest = 899 (Fable: 889).

## TICK (registered 2026-10-28 15:17 EDT, before any perturbed or rerun request): does one clock tick change a frontier model's action?
Motivation: E24 rerun divergence (11 of 28 first divergences had only status-bar differences). Same 1,000 AndroidControl steps and the same
16-shot prompt as AC-Fair (bench/ac_fewshot.py). Perturbation P (task-irrelevant, what changes between real reruns): the status-bar clock
advances one minute (text and description) and the battery percentage drops by one. No element is added, removed or reordered, so element
numbers and the gold answer are unchanged. Control R: the identical prompt sent again. Arms per model: A (original), R (identical rerun), P.
Models: claude-opus-5-5 (effort medium; A = the AC-Fair run) and claude-haiku-5-5 (effort low; A, R, P all new).
Flip = different action (type, element, text or app). H-T1 (per model): flips(A,P) > flips(A,R), paired exact McNemar on per-step flip
indicators, two-sided α = 0.05. H-T2 (per model): steps where A and R disagree are less often correct in A than steps where they agree
(Fisher exact). Also reported: accuracy of A, R, P; flip rates by action type.
Kill rule: if H-T1 fails for both models, we drop the claim that irrelevant screen changes alter frontier agents' decisions beyond sampling noise.

## AC-Fair amendment: effort confound (registered 2026-10-28 16:15 EDT, before the run; found by an independent review)
Zero-shot Opus (AC-Opus) ran at effort low; Opus-FS (AC-Fair) at effort medium. Arm Opus-ZS-med: claude-opus-5-5, effort medium, the
same SYSTEM prompt and screen rendering, NO examples (bench/ac_fewshot.py with ZS=1), same 1,000 steps. Decision rule: if Opus-ZS-med −
Opus-ZS-low ≥ half of (Opus-FS − Opus-ZS-low) = 46.5 steps, the AC-Fair conclusion "examples close the gap" is withdrawn as confounded;
otherwise examples are credited with the remainder (Opus-FS − Opus-ZS-med), reported with a paired McNemar test.
- **Effort amendment result (2026-10-28):** Opus-ZS-med 814 (vs ZS-low 800: +26/−12, p = 0.034). Effort effect 14 < 46.5, so the rule
  credits the examples: Opus-FS (893) − Opus-ZS-med (814) = +79 (+94/−15, p = 3.7e-15). Pointer − Opus-ZS-med = +6.4 (+120/−56, p = 1.6e-6).
- **TICK status (2026-10-28):** stopped after arms A and R for Haiku 5.5 (797, 790; 51 steps differ in correctness). The perturbed arm P and
  both Opus arms were never run: a literature check before spending found that sensitivity of LLMs and agents to irrelevant input changes is
  already documented, so the test could not produce a new result. No H-T1/H-T2 outcome is claimed.

## AC-Router (registered 2026-10-28 18:11 EDT, before any model sees the fresh steps): confirm the exploratory pointer→Opus cascade
Motivation: on bench/ac_sample1000.json the exploratory cascade (pointer on its most confident steps, Opus-FS on the rest) scored 895-900
for any local share from 50% to 90% vs Opus-FS 893. That threshold was chosen after seeing results, so it is tested here on fresh data.
Fresh steps: bench/ac_fresh1000.json = 1,000 AndroidControl TEST steps drawn (random.seed(20261028)) from the 836 test episodes with no step
in ac_sample1000. Frozen: pointer_v2 (outcomes already computed for all test steps), router rule and threshold in bench/ac_router_score.py
(THR = 0.9854, the 30th-percentile confidence on ac_sample1000; no change after this entry), Opus-FS harness bench/ac_fewshot.py unchanged
(claude-opus-5-5, effort medium, 16 retrieved TRAIN examples). Opus-FS runs on all 1,000 steps so the baseline and the cascade share calls.
H-AR1 (primary): router cascade non-inferior to Opus-FS alone, margin 2 points (lower 95% Wald bound of the paired difference > -2).
H-AR2: router cascade better than Opus-FS alone (paired exact McNemar, two-sided, alpha 0.05).
H-AR3: at least 60% of steps answered locally.
Also reported: pointer alone vs Opus-FS, cascade cost per 1k steps. Kill rule: if H-AR1 fails we drop "the cascade matches Opus at a fraction
of its calls" from every claim. Spend cap: $12.
- **AC-Router result (2026-10-28): H-AR1 SUPPORTED, H-AR2 not supported, H-AR3 SUPPORTED.** Fresh 1,000 steps: pointer 869, Opus-FS 882,
  router cascade 892. Cascade - Opus-FS = +1.0 [-0.2, +2.2] (+24/-14), McNemar p = 0.14: non-inferior at margin 2, not significantly better.
  70.7% of steps answered locally; on those steps the pointer is right 671/707 vs Opus-FS 661/707. Cascade vs pointer alone +2.3 (p = 0.011).
  Opus-FS cost $9.54 per 1k steps; cascade about $2.80 per 1k (29% of Opus calls; assumes deferred steps cost the average step).
  Batch msgbatch_01GDbv477aWMgUPQ9Rox6nXY, $9.54. The exploratory result on ac_sample1000 (900 vs 893) replicates in direction on fresh data.

## AC-Router-Sonnet (registered 2026-10-28 20:23 EDT, before Sonnet sees the fresh steps): does the router generalise to another frontier model?
Same 1,000 fresh steps (bench/ac_fresh1000.json), same frozen pointer, same router rule and THR = 0.9854 (bench/ac_router_score.py,
unchanged), same few-shot harness (bench/ac_fewshot.py, 16 retrieved TRAIN examples), model claude-sonnet-5-5, effort medium.
H-ARS1 (primary): pointer -> Sonnet-FS cascade non-inferior to Sonnet-FS alone, margin 2 points.
H-ARS2: pointer -> Sonnet-FS cascade non-inferior to Opus-FS alone (AC-Router run), margin 2 points.
Also reported: Sonnet-FS alone vs Opus-FS alone, measured cost from batch usage. Kill rule: if H-ARS1 fails, we state that routing was
confirmed with Opus only. Spend cap: $7.
Also recorded (no new spend, from the AC-Router batch's own usage): measured cascade cost $2.82 vs $9.54 per 1k steps (29.5%).
- **AC-Router-Sonnet result (2026-10-28): H-ARS1 SUPPORTED, H-ARS2 SUPPORTED.** Sonnet-FS alone 857; pointer -> Sonnet-FS cascade 881
  (+2.4 [+1.1, +3.7] vs Sonnet alone, +35/-11, McNemar p = 0.0005; superiority was not a registered hypothesis for this arm, reported as
  context). Cascade vs Opus-FS alone (882): -0.1 [-1.7, +1.5], p = 1.0, non-inferior at margin 2. Measured cost: Sonnet-FS $4.80 per 1k,
  cascade $1.41 per 1k = 29.5% of Sonnet and 14.8% of Opus-FS alone. Sonnet-FS is below Opus-FS (857 vs 882, p = 0.004).
  Batch msgbatch_01BxAomsUNsdweyfuoc9HVrg, $4.80. Routing generalises to a second frontier model with the threshold unchanged.

## AC-Router-Rerun (registered 2026-10-28 21:11 EDT, before the reruns): run-to-run stability of the routing result
A reviewer asked for repeated frontier runs. Arms: Opus-FS and Sonnet-FS rerun once each (RUNTAG _R2) on bench/ac_fresh1000.json with
everything unchanged (harness, examples, effort medium, router rule, THR = 0.9854). Reported: per-run and mean scores of each arm alone and
of each cascade; the range across the two runs; the registered AC-Router and AC-Router-Sonnet hypotheses re-tested on the run-2 outputs
and the mean count over the two runs. No hypothesis is changed. (Wording of this entry tidied before any rerun result existed.) Spend cap: $15.
- **AC-Router-Rerun result (2026-10-28):** run 2 (unchanged everything): Opus-FS 879 (run 1: 882), Sonnet-FS 862 (857). Cascades on run 2:
  pointer -> Opus 890 (+1.1 [-0.2, +2.4] vs Opus alone; H-AR1 non-inferiority SUPPORTED again), pointer -> Sonnet 889 (+2.7 [+1.3, +4.1]
  vs Sonnet alone; H-ARS1 SUPPORTED again) and +1.0 [-0.7, +2.7] vs Opus alone (H-ARS2 SUPPORTED again). Mean over runs: Opus 880.5,
  pointer -> Opus 891, Sonnet 859.5, pointer -> Sonnet 885. Each model's two runs differ in correctness on 21 of 1,000 steps. Spend $14.35.
- Exploratory, not registered (2026-10-28): AndroidControl-Curated labels (bench/ac_curated_score.py; 8,377 of 8,444 test steps match):
  pointer 92.0% on all matched steps (strict 88.6% on the same steps). Fresh steps (995 matched), run 1: pointer 909, Opus-FS 925,
  pointer -> Opus 928 (+0.3 [-0.8, +1.4]); Sonnet-FS 895, pointer -> Sonnet 914 (+1.9 [+0.7, +3.1]); pointer -> Sonnet vs Opus-FS -1.1
  [-2.5, +0.3] (would miss the 2-point margin). Alternative router scores (bench/ac_router_curves.py): margin and entropy match the
  registered max-probability score; action-type probability alone is worse (e.g. 862 vs 891 at 70% local with Opus).

## AC-Router-Haiku (registered 2026-10-28 21:44 EDT, before Haiku sees the fresh steps): a cheap fallback, effort-matched
Requested by a reviewer. Same fresh steps, frozen pointer, router rule and THR = 0.9854, same few-shot harness, model claude-haiku-5-5 at
effort medium (matching the Opus and Sonnet arms; the earlier Haiku 5.5 arm on ac_sample1000 ran at low effort).
H-ARH1: pointer -> Haiku-FS non-inferior to Haiku-FS alone, margin 2. H-ARH2: pointer -> Haiku-FS non-inferior to Opus-FS alone (run 1),
margin 2. Reported: costs from billed tokens. Spend cap: $1.50 (total budget nearly exhausted).
- **AC-Router-Haiku result (2026-10-28): H-ARH1 SUPPORTED, H-ARH2 NOT supported.** Haiku-FS (effort medium) alone 805; pointer -> Haiku
  865 (+6.0 [+4.3, +7.7] vs Haiku alone, p < 1e-12); vs Opus-FS alone -1.7 [-3.5, +0.1] (fails the 2-point margin); vs pointer alone
  -0.4 (p = 0.74): a fallback weaker than the pointer on hard steps adds nothing. Cost $0.26 per 1k for Haiku alone (Batch). Batch
  msgbatch_0112g2XY7rmoa1QPCuVN11Wx, $0.26. Total Anthropic spend $97.50.

## AC-LearnedRouter (registered 2026-10-29 00:01 EDT, before any model sees these steps): confirm the learned deferral rule
Exploratory result being confirmed: on ac_fresh1000, a learned rule (two logistic regressions fit on ac_sample1000: P(small model right)
and P(Opus-FS right); keep local when the difference is at least a fixed threshold) routed to Opus-FS scored 902 vs 882 for Opus alone.
Frozen before this entry: bench/results/learned_router_frozen.json (weights and threshold -0.01005, the value keeping 70.7% of ac_fresh1000
local), features from bench/ac_features.py (unchanged checkpoint pointer_v2), harness bench/ac_fewshot.py unchanged (Opus 5.5, effort
medium, 16 retrieved TRAIN examples).
Steps: bench/ac_confirm650.json = 650 test steps drawn (random.seed(20261029)) from the 257 test episodes used in neither ac_sample1000
nor ac_fresh1000 (the last unused test episodes).
H-LR1 (primary): learned-rule cascade better than Opus-FS alone (paired exact McNemar, two-sided, alpha 0.05).
H-LR2: learned-rule cascade non-inferior to Opus-FS alone, margin 2 points.
Also reported: the registered max-probability rule (THR 0.9854) on the same steps, local shares, cost. Spend cap: $7.
- **AC-LearnedRouter result (2026-10-29): H-LR1 SUPPORTED, H-LR2 SUPPORTED.** 650 steps (247 episodes): Opus-FS alone 581, small model alone
  581; learned-rule cascade 599 (73.7% local), +2.8 [+1.0, +4.5] vs Opus alone (+26/-8), McNemar p = 0.003. Registered max-probability
  rule on the same steps: 595 (73.5% local), +2.2 [+0.5, +3.9], p = 0.020. Learned vs registered rule: +0.6 [-0.7, +1.9], p = 0.48.
  Batch msgbatch_01RwgNVPwRbQxhBpmwu7NCNi, $6.15 (estimate).

## AC-Shortlist (registered 2026-10-29 13:40 EDT, before any frontier model sees these steps): the small model's top candidates as the fallback's input
Motivation: on deferred taps the small model's top 5 contains the target for 91% (held-out) and 90% (confirmatory) of steps, including
45 of 53 and 24 of 32 Opus errors (bench/ac_topk.py). The test split is exhausted, so this test uses fresh steps from the clean retrain's
validation episodes (TRAIN episodes with ep % 20 == 0, never trained on by the clean model).
Frozen before this entry (bench/sl_prepare.py, committed with this entry):
- Small model: the clean pruned-input retrain (training/kaggle_pointer-clean-r3, pointer_clean_r3.pt), pruned input as in training.
- Threshold: tau = 0.9842, the 30th percentile of the stored clean-model confidences on the calibration half (ep % 40 == 0, 1,691 steps).
- Sample: 1,000 steps drawn (random.seed(20261029)) from the test half (ep % 40 == 20, 1,804 steps); per-step row index,
  local decision and top-5 candidates in bench/results/sl_local.json (746 local, 254 deferred; small model alone 903).
- Frontier: claude-opus-5-5, effort medium, Batch API, same system prompt and 16 TF-IDF-retrieved examples as AC-Router; the retrieval
  pool excludes every validation episode. Harness bench/sl_frontier.py. Arms: full (all 1,000 steps, full screen); cand_full (deferred
  steps, full screen plus the top-5 candidate numbers); cand_only (deferred steps, only the 5 candidate elements shown).
- Disclosed: the sample comes from TRAIN-split episodes, so apps and task templates overlap with training (in-domain only).
Hypotheses (Holm correction over H-SL1 to H-SL3, family alpha 0.05; exact McNemar, two-sided, Wald 95% CI, on the 1,000 steps):
H-SL1: routing with cand_full better than routing with full (the candidate hint helps the fallback).
H-SL2: routing with cand_only non-inferior to routing with full, margin 2 points; reported with the mean screen tokens per deferred step.
H-SL3: routing with cand_full better than Opus full alone.
Secondary (no correction): routing with full non-inferior to Opus full alone, margin 2 (fresh-data replication of AC-Router); per-arm
accuracy on deferred taps; billed cost per arm. Estimated cost (count_tokens, Batch prices): full $10.27, cand_full $2.72, cand_only
$1.40. If only the deferred-step arms can be funded, the baseline runs as arm full_def (full screen, the 254 deferred steps only); H-SL1 and H-SL2 are then tested and H-SL3
is reported as not run. No arm, prompt or sample changes after the first batch is created.
