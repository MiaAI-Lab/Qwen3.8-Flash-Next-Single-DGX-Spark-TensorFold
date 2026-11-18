#!/usr/bin/env python3
"""scripts/harden.sh stays what it promised to be: opt-in, separate tag, no versions of its own, and gated on the server still booting.

The hardening layer from #77 is only safe while four things hold: nothing in the default build runs it, the image
`start.sh` uses is never rewritten, it installs no package version by itself (that is the operator's call, and a pin
in the script would be a dependency change nobody reviewed), and a hardened image that cannot boot is thrown away
rather than tagged. This checks all four offline, and that the README tells the truth about the CVE counts.

Reads scripts/harden.sh, scripts/prepare.sh, start.sh and README.md, and runs `bash -n` over the script. No Docker, no GPU.

Usage: tools/test_harden_recipe.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "scripts" / "harden.sh").exists():
    print("FAIL scripts/harden.sh: it is not there; scripts/harden.sh is what #77 asks for and what this checks")
    sys.exit(1)
HARDEN = (ROOT / "scripts" / "harden.sh").read_text(encoding="utf-8")
PREPARE = (ROOT / "scripts" / "prepare.sh").read_text(encoding="utf-8")
START = (ROOT / "start.sh").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")

failures: list[str] = []


def check(name: str, problems: list[str]) -> None:
    for p in problems:
        print(f"FAIL {name}: {p}")
        failures.append(name)
    if not problems:
        print(f"ok   {name}")


def value(name: str) -> str:
    m = re.search(rf'^{name}="\$\{{{name}:-([^"]*)\}}"', HARDEN, re.M)
    return m.group(1) if m else "\0missing"


# ---- it is a script that runs, and it is opt-in -------------------------------------------------

r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / "harden.sh")], capture_output=True, text=True)
check(
    "scripts/harden.sh parses and nothing in the default path runs it",
    ([] if r.returncode == 0 else [f"bash -n: {r.stderr.strip()}"])
    + ([] if (ROOT / "scripts" / "harden.sh").stat().st_mode & 0o111 else ["scripts/harden.sh is not executable"])
    + ([] if "harden.sh" not in PREPARE and "harden.sh" not in START else ["prepare.sh or start.sh calls harden.sh, so it is no longer opt-in"]),
)

# ---- the base image is left alone, the result gets its own tag ----------------------------------

check(
    "the hardened result is its own tag and $IMAGE survives",
    ([] if re.search(r'docker build [^\n]*-t "\$OUT"', HARDEN) else ["the build does not tag $OUT"])
    + ([] if not re.search(r'docker build[^\n]*-t "\$IMAGE"', HARDEN) else ["the build retags $IMAGE itself"])
    + ([] if not re.search(r'docker rmi[^\n]*\$IMAGE\b', HARDEN) else ["it removes $IMAGE, not just the hardened tag"])
    + ([] if re.search(r'docker rmi -f "\$OUT"', HARDEN) else ["a failed gate does not remove the hardened tag"])
    + ([] if re.search(r'tf\.base="\$IMAGE"', HARDEN) else ["the tag does not record the image it came from"]),
)

# ---- what it removes cannot reach what the server runs ------------------------------------------

lines = HARDEN.splitlines()
deletes = [m.group(1).strip() for ln in lines if not ln.lstrip().startswith("#") for m in [re.search(r"\brm -rf? ([^;|&]+)", ln)] if m]
check(
    "its deletions are only the CUDA-installed profilers and apt's lists",
    [d for d in deletes
     if not re.fullmatch(r"\$NSIGHT_GLOBS|/var/lib/apt/lists/\*|\$\{?NSIGHT_GLOBS\}?", d)]
    + ([] if "apt-get -y purge" in HARDEN else ["nothing is purged, so what does this script do"]),
)
purge = value("APT_PURGE")
check(
    "what it purges is the base image's ballast, never the engine's own files",
    ([] if "linux-libc-dev" in purge and "nsight" in purge else [f"APT_PURGE={purge!r} should name linux-libc-dev and the nsight packages"])
    + ([] if "/opt/tensorfold" not in HARDEN.replace("test -x /opt/tensorfold", "").replace("test -f /opt/tensorfold", "") or
       not re.search(r"(purge|rm -r[f]?)[^\n]*/opt/tensorfold", HARDEN) else ["a purge or deletion reaches /opt/tensorfold"])
    + ([] if re.search(r"install --reinstall python3-yaml[^\n]*\n[^\n]*rm -rf /var/lib/apt/lists", HARDEN) or
       HARDEN.find("install --reinstall python3-yaml") > HARDEN.find("autoremove --purge") else
       ["python3-yaml is not reinstalled after autoremove (the gotcha #77 hit)"]),
)

# ---- no versions of its own; the operator pins, or nobody upgrades --------------------------------

check(
    "it installs no package version of its own",
    ([] if value("PIP_UPGRADE") == "" else [f"PIP_UPGRADE defaults to {value('PIP_UPGRADE')!r}, it must default to nothing"])
    + ([] if value("PIP_UNINSTALL") == "" else [f"PIP_UNINSTALL defaults to {value('PIP_UNINSTALL')!r}, it must default to nothing"])
    + ([] if value("APT_UPGRADE") == "0" else [f"APT_UPGRADE defaults to {value('APT_UPGRADE')!r}, it must default to 0"])
    + ([] if not any(re.search(r"==\s*\d", ln.split(" #")[0]) for ln in lines if not ln.lstrip().startswith("#")) else ["the script pins a package version itself"])
    + ([] if "apt-get -y upgrade" not in HARDEN or 'if [ "$APT_UPGRADE" = 1 ]' in HARDEN else ["apt-get upgrade runs unconditionally"]),
)

# ---- the gate: what the hardened image has to still do before the tag is kept ---------------------

gate = HARDEN[HARDEN.find("CHECKS="):HARDEN.find("CHECKS=") + 1600]
check(
    "the boot gate covers what the recipe actually runs in the container",
    ([] if "tensorfold-native --help" in gate else ["the gate does not run tensorfold-native"])
    + ([] if "serve --help" in gate else ["the gate does not check the serve subcommand"])
    + ([] if "libturbojpeg" in gate and "libwebpdecoder" in gate and "libtfgrammar" in gate else ["the gate does not check the libraries the engine dlopens"])
    + ([] if re.search(r"import [^\"]*huggingface_hub", gate) else ["the gate does not check the Python the in-container download needs"])
    + ([] if "import yaml" in gate or re.search(r"import [^\"]*yaml", gate) else ["the gate does not check yaml (the autoremove gotcha)"])
    + ([] if "command -v hf" in gate else ["the gate does not check the hf CLI"]),
)

# ---- the README does not promise a CVE count -------------------------------------------------------

check(
    "README says what hardening is, and does not promise CVE counts",
    ([] if "scripts/harden.sh" in README else ["the README never mentions scripts/harden.sh"])
    + ([] if "APT_UPGRADE" in README and "PIP_UPGRADE" in README else ["the README does not document the two opt-in knobs"])
    + ([] if re.search(r"[Nn]o scanner|no scanner", README) else ["the README should say no scanner was run here"])
    + ([] if "does not get smaller" in README else ["the README does not say the image does not shrink"])
    + ([] if re.search(r"-hardened-", README) else ["the README does not say the result is a separate tag"]),
)

if failures:
    print(f"\n{len(failures)} hardening check(s) failed")
    sys.exit(1)
print("\nall passed")
