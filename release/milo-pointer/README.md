---
license: apache-2.0
base_model: answerdotai/ModernBERT-base
tags: [gui-agent, android, accessibility, on-device, androidcontrol]
---
# Milo Pointer (149M)

Paper: *A 149M Text Encoder as the First Tier for Phone Agents: Pre-Registered Routing to Frontier Models*
(Chaitanya Talapalli, 2026; [doi:10.5281/zenodo.23818333](https://doi.org/10.5281/zenodo.23818333); ORCID [0009-0000-8811-3349](https://orcid.org/0009-0000-8811-3349)). Code, per-step results and the registration log: https://github.com/chayworks/milo-pointer

- **A 149M text-only model scores 86.1% on AndroidControl-Low** under the protocol screenshot agents report, above the 85.2%
  reported for OS-Atlas-7B (step sets differ slightly).
- **Small model + frontier beats the frontier alone.** The small model answers about 70% of steps on the device and hands the
  rest to Claude Opus 5.5. In a pre-registered confirmatory test routing beat Opus alone by 2.8 points (p = 0.003); pooled over
  all 1,650 test steps the gain is 1.5 points (95% CI [+0.5, +2.4]), at under a third of Opus's API cost.
- **What did not work is documented too.** About 40 registered experiments, negative ones included, with timestamps
  (`paper/PREREGISTRATION.md`).

All Claude API calls behind the AndroidControl results cost **$86.75** in total (Batch prices); each training run takes about
**6.4 hours on one NVIDIA T4**.

![Cost and accuracy on 1,000 held-out steps; routing curves](https://huggingface.co/coolnow/milo-pointer/resolve/main/fig_router.png)

Demo: the model drives a Pixel 6a through a five-step Clock task ("set a 5 minute timer"), every step answered locally
([video](https://huggingface.co/coolnow/milo-pointer/resolve/main/milo_demo_clean.mp4)).

A 149M-parameter model that picks the next action on an Android screen from its accessibility tree, in about 0.3 s on a laptop.
It is meant to sit **in front of** a frontier agent: it answers the steps it is sure about and returns `defer: true` for the rest.

## Result (pre-registered, fresh data)

1,000 AndroidControl test steps (episodes not used in any earlier frontier comparison), each with a human-written step instruction;
offline, from the recorded history. The routing threshold was fixed on a separate sample and committed before Claude saw these steps
(`paper/PREREGISTRATION.md`, AC-Router). The pointer itself was selected on the test split during development, so its own score is
mildly optimistic. Costs: billed tokens, Batch API prices (50% off); double them for standard real-time calls.

| System | Run 1 | Run 2 | Steps on device | API $ per 1k steps (Batch) |
|---|---|---|---|---|
| Milo Pointer alone | 869 | 869 | 100% | 0 |
| Claude Opus 5.5 + 16 retrieved examples | 882 | 879 | 0% | 9.54 |
| **Milo Pointer, defer to Opus 5.5** | **892** | **890** | **70.7%** | **2.82** |
| Claude Sonnet 5.5 + 16 retrieved examples | 857 | 862 | 0% | 4.80 |
| **Milo Pointer, defer to Sonnet 5.5** | **881** | **889** | **70.7%** | **1.41** |

A further registered test on the last 650 unused test steps, with a learned hand-off rule frozen in advance, found routing to Opus
**more accurate than Opus alone: 599 vs 581 (+2.8 points, 95% CI [+1.0, +4.5], p = 0.003)**, with 73.7% of steps kept local.

On the 1,000 steps above, the cascade is non-inferior to Opus alone (+1.0 points, 95% CI [-0.2, +2.2]; not significantly better, p = 0.14) and uses 29% of
its calls. On the steps the pointer keeps it is right 94.9% of the time, against 93.5% for Opus on the same steps.

Full AndroidControl test set (8,444 steps, strict metric): 88.5% step accuracy; IDD 89.8, unseen app 87.9, unseen task 87.2,
unseen category 86.7.

What it does **not** do: it is not better than Opus given the same data (878 vs 893 on another sample, p = 0.17). Live on
AndroidWorld, a cascade of this kind (deferring to Gemini 3.5 Flash) completed fewer tasks than the frontier agent alone (41.9% vs
54.8%, p = 0.008): offline step accuracy does not carry over to live tasks where errors compound. Both the pointer and the few-shot
Claude arms use AndroidControl's 74k labelled training steps; on your own apps expect to need similar data.

## Use

```python
# pip install torch transformers safetensors huggingface_hub
from huggingface_hub import snapshot_download
path = snapshot_download("coolnow/milo-pointer")
import sys; sys.path.insert(0, path)
from milo_pointer import MiloPointer
m = MiloPointer.load(path)
a = m.predict(goal="Turn on dark mode", instruction="Open display settings",
              elements=[{"cls": "TextView", "text": "Display", "b": [0, 400, 1080, 520], "click": 1}, ...],
              width=1080, height=2400, last_step="new step")
if a["defer"]:
    ...  # send this step to your frontier agent
else:
    ...  # a = {'action_type': 'click', 'element': 3, 'x': 540, 'y': 460, 'confidence': 0.998, 'defer': False}
```

Actions: click, long_press, input_text (text copied from the instruction), scroll_up/down/left/right, navigate_back,
navigate_home, open_app (name copied from the instruction), wait. `last_step` describes the previous step; see
`bench/ac_frontier.py:test_rows` for the format used in training ("new step", or "repeat of last step (last action click), screen
80% unchanged").

## Speed and memory

| Hardware | Median per step | Memory |
|---|---|---|
| Apple M2 laptop GPU (PyTorch, fp16) | 0.29 s | ~2 GB process |
| Pixel 6a CPU, ONNX Runtime 1.20, 4 threads, fp32 | 4.5 s (p90 11.9 s) | 1.94 GB |
| Pixel 6a CPU, ONNX Runtime 1.20, 4 threads, int8 | 3.4 s (p90 11.2 s) | 1.45 GB (agrees with fp32 on 34/40 steps) |
| Same, empty layout containers removed from the input | 1.18 s (p90 4.9 s) | 1.45 GB (0.8 points lower accuracy, routing intact) |

int8 accuracy on 1,000 held-out AndroidControl steps (fp32: 869): per-tensor `pointer_int8.onnx` 851, per-channel `onnx/pointer_int8_pc.onnx` (151 MB, recommended; built by `android/quant_variants.py`) 862. Per-channel was chosen on a separate calibration sample. Phone latencies above are for the per-tensor model; on the phone the per-channel model runs within 5% of it.

Phone CPU latency is not yet sub-second. `android/` has the export, the benchmark and a live demo.

## Limits

Text only (no screenshot), so icons without labels are hard. Trained on AndroidControl only; expect lower accuracy on very
different apps (on a strict holdout of 72 unseen apps, 86.2 to 82.6). English instructions. Frontier comparisons are single runs.

## Training

ModernBERT-base encoder, three heads (action type, element pointer, copy span), 2 epochs on the 74,714 AndroidControl TRAIN steps,
a single Kaggle GPU session. Source: `training/ac/pointer_v2/src/v2.py`.
