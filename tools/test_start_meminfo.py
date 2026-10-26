#!/usr/bin/env python3
"""start.sh reads the available memory without depending on the locale (issue #37).

`free -g` prints translated row labels (`Speicher:`, `内存：`), so an awk match on `^Mem:` came back empty on
non-English systems and start.sh refused to start. This runs start.sh's own `avail_gb=` line with a `free` on PATH
that prints a German row, and checks the figure still comes out as MemAvailable in whole GiB.

Usage: tools/test_start_meminfo.py        (needs bash and /proc/meminfo; no GPU, no Docker)
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def meminfo_gib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1048576
    raise SystemExit("no MemAvailable in /proc/meminfo")


def main() -> int:
    lines = [ln for ln in (ROOT / "start.sh").read_text().splitlines() if ln.startswith("avail_gb=")]
    if len(lines) != 1:
        print(f"FAIL: expected one avail_gb= line in start.sh, found {len(lines)}")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        free = Path(tmp) / "free"
        free.write_text("#!/bin/sh\nprintf '%s\\n' '               gesamt   benutzt   frei  gemns.  Puffer/Cache  verfügbar'"
                        " 'Speicher:      121        20       30       0        70         95'\n")
        free.chmod(0o755)
        env = {**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}", "LC_ALL": "C"}
        before = meminfo_gib()
        out = subprocess.run(["bash", "-c", lines[0] + '\nprintf %s "$avail_gb"'], capture_output=True, text=True,
                             env=env).stdout.strip()
        after = meminfo_gib()
    if not re.fullmatch(r"\d+", out):
        print(f"FAIL: avail_gb is {out!r}, not a number (localized free output was not parsed)")
        return 1
    if not min(before, after) <= int(out) <= max(before, after):
        print(f"FAIL: avail_gb {out} is not MemAvailable ({before}..{after} GiB)")
        return 1
    print(f"ok: avail_gb {out} GiB = MemAvailable with a German `free`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
