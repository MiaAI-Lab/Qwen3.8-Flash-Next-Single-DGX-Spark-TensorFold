#!/usr/bin/env python3
"""The README and start.sh's own usage say what this build actually serves.

Offline: reads README.md, start.sh and scripts/prepare.sh from the repository.

Two claims drift and have been reported:
  * #89: the docs offered a 1,048,576-token window with YaRN factor 4 while `start.sh` refuses any
    `CONTEXT` above the engine's native 262,144 (the Flash Next CUDA family refuses a scaled rope).
  * the CUDA engine reads no video (the Flash Next family has no video path), while the vision
    section listed video formats, a per-request video budget and a measured video timing.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
START = (ROOT / "start.sh").read_text(encoding="utf-8")
PREPARE = (ROOT / "scripts" / "prepare.sh").read_text(encoding="utf-8")

failures: list[str] = []


def check(name: str, problems: list[str]) -> None:
    if problems:
        for p in problems:
            print(f"FAIL {name}: {p}")
        failures.append(name)
    else:
        print(f"ok   {name}")


def window_problems(where: str, text: str, cap: int) -> list[str]:
    """Lines that offer a context above the engine's cap, or a YaRN that would lift it."""
    bad = []
    for at, line in enumerate(text.splitlines(), 1):
        guard_line = re.search(r"=~|die \"", line)  # a validation guard names the bound without offering it
        number = re.search(r"1[ ,]?048[ ,]?576", line)
        if number and not guard_line and re.search(r"\bCONTEXT\b", line, re.I):
            bad.append(f"{at}: offers {number.group(0)} as a CONTEXT setting: {line.strip()}")
        if re.search(r"\bYaRN\b", line) and not re.search(r"no YaRN|refus|not supported", line, re.I):
            bad.append(f"{at}: advertises YaRN without saying this engine has none: {line.strip()}")
    return bad


guard = re.search(r"\(\(\s*CONTEXT\s*<=\s*(\d+)\s*\)\)", START)
check("start.sh caps CONTEXT", [] if guard else ["no '(( CONTEXT <= N ))' guard found"])
cap = int(guard.group(1)) if guard else 0
check("the cap is the engine's native 262,144", [] if cap == 262144 else [f"the guard says {cap}"])
check(f"README.md keeps the window at {cap}", window_problems("README.md", README, cap))
check(f"start.sh keeps the window at {cap}", window_problems("start.sh", START, cap))

# ---- images and video ---------------------------------------------------------------------

serves_video = bool(re.search(r"zig build[^\n]*\bvideo\b", PREPARE)) or "libtfvideo" in PREPARE
if serves_video:
    print("ok   this build links a video decoder: the video checks do not apply")
else:
    section = re.search(r"^## Images and video\b.*?(?=^## )", README, re.M | re.S)
    check("README has an 'Images and video' section", [] if section else ["section not found"])
    body = section.group(0) if section else ""
    check(
        "the vision section says a video_url part is refused",
        [] if re.search(r"video[^\n]*(?:refus|not served|unsupported|not accepted)", body, re.I)
        else ["no line says the engine refuses video input"],
    )
    check(
        "the vision section quotes the server's own refusal",
        [] if "audio and video inputs are unsupported" in body else ["the refusal text is not the engine's"],
    )
    check(
        "the vision section lists no video formats",
        [l.strip() for l in body.splitlines() if "FFmpeg decodes" in l or "MP4, WebM" in l],
    )
    check(
        "the vision section quotes no video timing",
        [l.strip() for l in body.splitlines() if re.search(r"\d+-second video", l)],
    )
    check(
        "the vision section says the engine decodes images itself",
        [l.strip() for l in body.splitlines() if "Pillow" in l or "PyAV" in l],
    )

if failures:
    print(f"\n{len(failures)} claim(s) in the docs do not match this build")
    sys.exit(1)
print("\nall passed")
