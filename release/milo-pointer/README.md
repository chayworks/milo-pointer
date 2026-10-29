---
license: apache-2.0
base_model: answerdotai/ModernBERT-base
tags: [gui-agent, android, accessibility, on-device, androidcontrol]
---
# Milo Pointer (149M)

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

The cascade is non-inferior to Opus alone (+1.0 points, 95% CI [-0.2, +2.2]; not significantly better, p = 0.14) and uses 29% of
its calls. On the steps the pointer keeps it is right 94.9% of the time, against 93.5% for Opus on the same steps.

Full AndroidControl test set (8,444 steps, strict metric): 88.5% step accuracy; IDD 89.8, unseen app 87.9, unseen task 87.2,
unseen category 86.7.

What it does **not** do: it is not better than Opus given the same data (878 vs 893 on another sample, p = 0.17). Live on
AndroidWorld, a cascade of this kind (deferring to Gemini 3.5 Flash) completed fewer tasks than the frontier agent alone (41.9% vs
54.8%, p = 0.008): offline step accuracy does not carry over to live tasks where errors compound. Both the pointer and the few-shot
Claude arms use AndroidControl's 74k labelled training steps; on your own apps expect to need similar data.

## Use

```python
from milo_pointer import MiloPointer
m = MiloPointer.load("milo-pointer")
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

Phone CPU latency is too slow for interactive use today. `android/` has the export, the benchmark and a live demo.

## Limits

Text only (no screenshot), so icons without labels are hard. Trained on AndroidControl only; expect lower accuracy on very
different apps (on a strict holdout of 72 unseen apps, 86.2 to 82.6). English instructions. Frontier comparisons are single runs.

## Training

ModernBERT-base encoder, three heads (action type, element pointer, copy span), 2 epochs on the 74,714 AndroidControl TRAIN steps,
a single Kaggle GPU session. Source: `training/ac/pointer_v2/src/v2.py`.
