"""Read the connected Android phone's current screen (uiautomator accessibility dump over adb) into Milo Pointer elements,
and perform actions. Only for apps without accounts (the demo uses Clock and Settings)."""

import re, subprocess, xml.etree.ElementTree as ET
from pathlib import Path

ADB = str(Path.home() / "tools/platform-tools/adb")


def sh(*a, timeout=30):
    return subprocess.run([ADB, *a], capture_output=True, text=True, timeout=timeout).stdout


def screen():
    sh("shell", "uiautomator", "dump", "/sdcard/milo_ui.xml")
    xml = sh("shell", "cat", "/sdcard/milo_ui.xml")
    root = ET.fromstring(xml[xml.index("<") :])
    els = []
    for n in root.iter("node"):
        b = [int(v) for v in re.findall(r"\d+", n.get("bounds", "[0,0][0,0]"))]
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        rid = n.get("resource-id", "")
        rid = rid.split(":id/")[-1] if ":id/" in rid else rid
        f = lambda k: int(n.get(k) == "true")
        els.append(
            {
                "cls": n.get("class", "").split(".")[-1],
                "text": n.get("text", ""),
                "desc": n.get("content-desc", ""),
                "hint": n.get("hint", "") or "",
                "rid": rid,
                "pkg": n.get("package", ""),
                "b": b,
                "click": f("clickable"),
                "long": f("long-clickable"),
                "edit": int(n.get("class", "").endswith("EditText")),
                "scroll": f("scrollable"),
                "checked": f("checked"),
                "sel": f("selected"),
                "focus": f("focused"),
            }
        )
    w, h = map(int, re.findall(r"\d+", sh("shell", "wm", "size"))[-2:])
    return els, w, h


def act(a, els, w, h):
    t = a["action_type"]
    if t in ("click", "long_press"):
        x, y = a["x"], a["y"]
        sh(
            "shell",
            "input",
            "tap" if t == "click" else "swipe",
            *(map(str, (x, y) if t == "click" else (x, y, x, y, 800)))
        )
    elif t == "input_text":
        sh("shell", "input", "text", a["text"].replace(" ", "%s"))
    elif t.startswith("scroll_"):
        d = t[7:]
        cx, cy = w // 2, h // 2
        dx, dy = {"down": (0, -h // 3), "up": (0, h // 3), "right": (-w // 3, 0), "left": (w // 3, 0)}[d]
        sh("shell", "input", "swipe", str(cx), str(cy), str(cx + dx), str(cy + dy), "300")
    elif t == "navigate_back":
        sh("shell", "input", "keyevent", "4")
    elif t == "navigate_home":
        sh("shell", "input", "keyevent", "3")
    elif t == "open_app":
        pk = {
            "clock": "com.google.android.deskclock",
            "settings": "com.android.settings",
            "calculator": "com.google.android.calculator",
        }
        p = pk.get(a["app_name"].strip().lower())
        if p:
            sh("shell", "monkey", "-p", p, "-c", "android.intent.category.LAUNCHER", "1")
        else:
            print("  (open_app refused: not on the safe list)", a["app_name"])
