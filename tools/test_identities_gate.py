#!/usr/bin/env python3
"""start.sh does not offer an identity (a refusal-removed model) on an open port.

With TENSORFOLD_IDENTITIES (ABLIT=2 sets it) or TENSORFOLD_ABLATE set, start.sh needs an API key (TENSORFOLD_API_KEY,
--api-key, --api-key-file) or a loopback host. This runs config.sh's own identities_unguarded over the cases, then
start.sh itself with ABLIT=2 on 0.0.0.0 and no key (it must refuse before touching Docker), and with a key (it must not).
start.sh runs with DRY_RUN=1 and a `docker` on PATH that only fails, so nothing is started or stopped.

Usage: tools/test_identities_gate.py        (needs bash; no GPU, no Docker, no network)
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFUSAL = "safety refusals have been removed"


def function() -> str:
    text = (ROOT / "scripts/config.sh").read_text()
    m = re.search(r"^identities_unguarded\(\) \{.*?^\}\n", text, re.S | re.M)
    if not m:
        raise SystemExit("FAIL: identities_unguarded not found in scripts/config.sh")
    return m.group(0)


def unguarded(env: dict, host: str, *args: str) -> bool:
    call = " ".join(["identities_unguarded", host, *args])
    script = "set -u\n" + function() + call + "\n"
    clean = {k: v for k, v in os.environ.items() if not k.startswith("TENSORFOLD_")}
    return subprocess.run(["bash", "-c", script], env={**clean, **env}, capture_output=True).returncode == 0


def start(env: dict, *args: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as bin_dir:
        stub = Path(bin_dir, "docker")
        stub.write_text("#!/bin/sh\nexit 1\n")
        stub.chmod(0o755)
        keep = {k: v for k, v in os.environ.items() if k in ("HOME", "LANG", "LC_ALL", "TERM")}
        base = {**keep, "PATH": f"{bin_dir}:{os.environ['PATH']}", "DRY_RUN": "1", "CHECKPOINT": "nvfp4",
                "HOST": "0.0.0.0", "TENSORFOLD_API_KEY": "", "TENSORFOLD_IDENTITIES": "", "TENSORFOLD_ABLATE": "",
                "STATE_DIR": str(Path(bin_dir, "state")), "KERNEL_CACHE": str(Path(bin_dir, "kc")), "HF_CACHE": bin_dir}
        return subprocess.run(["bash", str(ROOT / "start.sh"), *args], env={**base, **env}, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=120)


def main() -> int:
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f": got {got!r}, want {want!r}"))

    ident = {"TENSORFOLD_IDENTITIES": "m-abliterated=/root/.cache/huggingface/ablate/a.safetensors"}
    check("no identity: never refused", unguarded({}, "0.0.0.0"), False)
    check("an identity on 0.0.0.0 with no key: refused", unguarded(ident, "0.0.0.0"), True)
    check("an identity on :: with no key: refused", unguarded(ident, "::"), True)
    check("an identity on a LAN address with no key: refused", unguarded(ident, "192.168.0.10"), True)
    check("TENSORFOLD_ABLATE alone counts as an identity", unguarded({"TENSORFOLD_ABLATE": "/a.safetensors"}, "0.0.0.0"), True)
    check("a blank TENSORFOLD_API_KEY is no key", unguarded({**ident, "TENSORFOLD_API_KEY": "  "}, "0.0.0.0"), True)
    check("TENSORFOLD_API_KEY set: allowed", unguarded({**ident, "TENSORFOLD_API_KEY": "sk-x"}, "0.0.0.0"), False)
    check("--api-key among the serve args: allowed", unguarded(ident, "0.0.0.0", "--parallel", "2", "--api-key", "k"), False)
    check("--api-key=k: allowed", unguarded(ident, "0.0.0.0", "--api-key=k"), False)
    check("--api-key-file: allowed", unguarded(ident, "0.0.0.0", "--api-key-file", "/k"), False)
    check("--api-key= with nothing after it is no key", unguarded(ident, "0.0.0.0", "--api-key="), True)
    for host in ("127.0.0.1", "127.0.1.1", "localhost", "::1"):
        check(f"loopback {host} with no key: allowed", unguarded(ident, host), False)

    refused = start({"ABLIT": "2"})
    check("start.sh ABLIT=2 on 0.0.0.0, no key: exits 1 with the reason",
          (refused.returncode, REFUSAL in refused.stderr), (1, True))
    moved = start({"ABLIT": "2"}, "--host", "127.0.0.1")
    check("start.sh ABLIT=2 with --host 127.0.0.1 among its args: not refused", REFUSAL in moved.stderr, False)
    opened = start({"ABLIT": "2", "HOST": "127.0.0.1"}, "--host", "0.0.0.0")
    check("start.sh ABLIT=2, HOST=127.0.0.1 but --host 0.0.0.0 among its args: refused", REFUSAL in opened.stderr, True)
    keyed = start({"ABLIT": "2", "TENSORFOLD_API_KEY": "sk-test"})
    check("start.sh ABLIT=2 with TENSORFOLD_API_KEY: not refused", REFUSAL in keyed.stderr, False)
    stock = start({"ABLIT": "0"})
    check("start.sh ABLIT=0, no key: not refused", REFUSAL in stock.stderr, False)

    print("FAIL" if fails else "all passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
