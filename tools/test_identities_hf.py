#!/usr/bin/env python3
"""TENSORFOLD_IDENTITIES entries of the form id=hf:owner/repo/path[@revision][:sha256:HEX] (scripts/config.sh).

Runs config.sh's own hf_identities .. identities_resolved against a fake Hugging Face cache and checks that hf: entries
become the file's path inside the container, plain paths pass through, a revision is honoured, a pinned SHA-256 is
checked, and an entry whose file is not downloaded, or does not match its pin, fails instead of reaching the engine.
Also checks that ABLIT=2's directions file is recorded like the other pinned downloads: repo, full commit and SHA-256.

Usage: tools/test_identities_hf.py        (needs bash; no GPU, no Docker, no network)
"""
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHA = "0123456789abcdef0123456789abcdef01234567"
PIN = "89abcdef0123456789abcdef0123456789abcdef"
CONTENT = b"x"
DIGEST = hashlib.sha256(CONTENT).hexdigest()
OTHER = hashlib.sha256(b"y").hexdigest()


def functions() -> str:
    text = (ROOT / "scripts/config.sh").read_text()
    m = re.search(r"^hf_identities\(\) \{.*?^identities_resolved\(\) \{.*?^\}\n", text, re.S | re.M)
    if not m:
        raise SystemExit("FAIL: hf_identities .. identities_resolved not found in scripts/config.sh")
    return m.group(0)


def run(cache: str, identities: str, call: str) -> subprocess.CompletedProcess:
    script = "set -u\n" + functions() + f"\nHF_CACHE={cache!r}\nTENSORFOLD_IDENTITIES={identities!r}\n{call}\n"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def main() -> int:
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f": got {got!r}, want {want!r}"))

    def refused(name, proc, reason):
        check(name, (proc.returncode != 0, proc.stdout, reason in proc.stderr), (True, "", True))

    with tempfile.TemporaryDirectory() as cache:
        repo = Path(cache, "hub/models--me--dirs")
        (repo / "refs").mkdir(parents=True)
        (repo / "refs/main").write_text(SHA)
        for snap in (SHA, PIN):
            (repo / "snapshots" / snap).mkdir(parents=True)
            (repo / "snapshots" / snap / "a.safetensors").write_bytes(CONTENT)
        inside = "/root/.cache/huggingface/hub/models--me--dirs/snapshots"

        listed = ("abl=hf:me/dirs/a.safetensors, local=/root/.cache/huggingface/ablate/b.safetensors,"
                  f"pinned=hf:me/dirs/a.safetensors@{PIN}:sha256:{DIGEST}")
        check("hf_identities lists the hf: entries, with the pinned SHA-256 or -", run(cache, listed, "hf_identities").stdout,
              f"me/dirs main a.safetensors -\nme/dirs {PIN} a.safetensors {DIGEST}\n")
        check("identities_resolved rewrites hf: entries and passes plain ones through unchanged",
              run(cache, listed, "identities_resolved").stdout.strip(),
              f"abl={inside}/{SHA}/a.safetensors, local=/root/.cache/huggingface/ablate/b.safetensors,pinned={inside}/{PIN}/a.safetensors")
        check("a pin with no revision reads main", run(cache, f"p=hf:me/dirs/a.safetensors:sha256:{DIGEST}", "hf_identities").stdout,
              f"me/dirs main a.safetensors {DIGEST}\n")
        check("no hf: entry: nothing listed", run(cache, "x=/a.safetensors", "hf_identities").stdout, "")

        refused("a file that is not downloaded fails", run(cache, "abl=hf:me/dirs/missing.safetensors", "identities_resolved"),
                "is not downloaded")
        refused("a file whose SHA-256 is not the pinned one fails",
                run(cache, f"abl=hf:me/dirs/a.safetensors@{PIN}:sha256:{OTHER}", "identities_resolved"), f"not the pinned {OTHER}")
        refused("a malformed pin fails", run(cache, "abl=hf:me/dirs/a.safetensors:sha256:abc", "identities_resolved"),
                "64 lowercase hex digits")
        check("hf:owner/repo with no file lists ? as its path", run(cache, "abl=hf:me/dirs", "hf_identities").stdout, "me/dirs main ? -\n")
        refused("hf:owner/repo with no file fails", run(cache, "abl=hf:me/dirs", "identities_resolved"), "names no file")

    config = (ROOT / "scripts/config.sh").read_text()
    pins = {
        "ABLIT_DIRECTIONS_ID": r"ABLIT_DIRECTIONS_ID=[\w.-]+/[\w.-]+",
        "ABLIT_DIRECTIONS_REVISION": r'ABLIT_DIRECTIONS_REVISION="\$\{ABLIT_DIRECTIONS_REVISION:-[0-9a-f]{40}\}"',
        "ABLIT_DIRECTIONS_FILE": r"ABLIT_DIRECTIONS_FILE=[\w./-]+\.safetensors",
        "ABLIT_DIRECTIONS_SHA256": r'ABLIT_DIRECTIONS_SHA256="\$\{ABLIT_DIRECTIONS_SHA256:-[0-9a-f]{64}\}"',
    }
    for name, pattern in pins.items():
        check(f"config.sh pins {name}", bool(re.search(f"^{pattern}$", config, re.M)), True)
    check("ABLIT=2's entry carries the pinned revision and SHA-256",
          "=hf:$ABLIT_DIRECTIONS_ID/$ABLIT_DIRECTIONS_FILE@$ABLIT_DIRECTIONS_REVISION:sha256:$ABLIT_DIRECTIONS_SHA256\"" in config, True)

    print("FAIL" if fails else "all passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
