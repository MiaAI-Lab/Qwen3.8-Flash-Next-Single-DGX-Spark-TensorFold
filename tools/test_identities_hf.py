#!/usr/bin/env python3
"""TENSORFOLD_IDENTITIES entries of the form id=hf:owner/repo/path[@revision] (scripts/config.sh).

Runs config.sh's own hf_identities, hf_snapshot and identities_resolved against a fake Hugging Face cache and checks
that hf: entries become the file's path inside the container, plain paths pass through, a revision is honoured, and
an entry whose file is not downloaded fails instead of passing a dead path to the engine.

Usage: tools/test_identities_hf.py        (needs bash; no GPU, no Docker, no network)
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHA = "0123456789abcdef0123456789abcdef01234567"
PIN = "89abcdef0123456789abcdef0123456789abcdef"


def functions() -> str:
    text = (ROOT / "scripts/config.sh").read_text()
    m = re.search(r"^hf_identities\(\) \{.*?^identities_resolved\(\) \{.*?^\}\n", text, re.S | re.M)
    if not m:
        raise SystemExit("FAIL: hf_identities .. identities_resolved not found in scripts/config.sh")
    return m.group(0)


def run(cache: str, identities: str, call: str) -> subprocess.CompletedProcess:
    script = functions() + f'\nHF_CACHE={cache!r}\nTENSORFOLD_IDENTITIES={identities!r}\n{call}\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def main() -> int:
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f": got {got!r}, want {want!r}"))

    with tempfile.TemporaryDirectory() as cache:
        repo = Path(cache, "hub/models--me--dirs")
        (repo / "refs").mkdir(parents=True)
        (repo / "refs/main").write_text(SHA)
        for snap in (SHA, PIN):
            (repo / "snapshots" / snap).mkdir(parents=True)
            (repo / "snapshots" / snap / "a.safetensors").write_bytes(b"x")
        inside = "/root/.cache/huggingface/hub/models--me--dirs/snapshots"

        listed = "abl=hf:me/dirs/a.safetensors, local=/root/.cache/huggingface/ablate/b.safetensors,pinned=hf:me/dirs/a.safetensors@" + PIN
        check("hf_identities lists the hf: entries", run(cache, listed, "hf_identities").stdout,
              f"me/dirs main a.safetensors\nme/dirs {PIN} a.safetensors\n")
        check("identities_resolved rewrites hf: entries and passes plain ones through unchanged",
              run(cache, listed, "identities_resolved").stdout.strip(),
              f"abl={inside}/{SHA}/a.safetensors, local=/root/.cache/huggingface/ablate/b.safetensors,pinned={inside}/{PIN}/a.safetensors")
        check("no hf: entry: nothing listed", run(cache, "x=/a.safetensors", "hf_identities").stdout, "")
        missing = run(cache, "abl=hf:me/dirs/missing.safetensors", "identities_resolved")
        check("a file that is not downloaded fails", (missing.returncode != 0, missing.stdout), (True, ""))
        check("hf:owner/repo with no file lists an empty path", run(cache, "abl=hf:me/dirs", "hf_identities").stdout, "me/dirs main \n")
        nofile = run(cache, "abl=hf:me/dirs", "identities_resolved")
        check("hf:owner/repo with no file fails", (nofile.returncode != 0, nofile.stdout), (True, ""))

    print("FAIL" if fails else "all passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
