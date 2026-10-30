"""Live demo: the 149M pointer drives a real Pixel 6a through a Clock task from step instructions. The model runs locally on the
laptop next to the phone (no cloud calls); the phone is read and driven over adb. Steps the model is unsure of are flagged as
'defer' (in deployment they would go to a frontier model); this demo then stops rather than guess.
Usage (repo root): python release/milo-pointer/android/demo.py"""

import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import screen
from milo_pointer import MiloPointer

GOAL = "Set a 5 minute timer in the Clock app"
STEPS = ["Go to the Timers tab", "Tap 5", "Tap 0", "Tap 0", "Start the timer"]
m = MiloPointer.load(HERE.parent)
log = []
last = None
m.predict(GOAL, "warm up", *screen.screen()[:1], 1080, 2400)  # first call compiles kernels; not timed
import subprocess

rec = subprocess.Popen([screen.ADB, "shell", "screenrecord", "--time-limit", "30", "/sdcard/milo_demo.mp4"])
time.sleep(1.5)
for ins in STEPS:
    els, w, h = screen.screen()
    t0 = time.time()
    a = m.predict(GOAL, ins, els, w, h, last_step="new step" + (f" (last action {last})" if last else ""))
    ms = 1000 * (time.time() - t0)
    tgt = els[a["element"]] if "element" in a else None
    desc = (tgt["text"] or tgt["desc"] or tgt["rid"]) if tgt else a.get("text") or a.get("app_name") or ""
    print(
        f"{ins:24} -> {a['action_type']:10} {desc[:28]:28} conf {a['confidence']:.3f} {'DEFER' if a['defer'] else ''} {ms:.0f} ms",
        flush=True,
    )
    log.append({"instruction": ins, "action": a, "target": desc, "model_ms": round(ms)})
    if a["defer"]:
        print("model unsure: stopping (would ask the cloud model)")
        break
    screen.act(a, els, w, h)
    last = a["action_type"]
    time.sleep(1.5)
time.sleep(2)
rec.wait()
json.dump({"goal": GOAL, "steps": log}, open(HERE / "demo_log.json", "w"), indent=1)
