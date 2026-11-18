#!/usr/bin/env python3
"""PDF document parts are opt-in, and the image carries a checksummed PDFium (issue #88, PDF=1).

The engine reads PDF parts only with `--vision-pdf`, and only if it can load a PDFium; the recipe pins it out of the
pypdfium2 aarch64 wheel, so a pin that no longer matches the wheel, a build that stops checking it, or a `--vision-pdf`
that reaches the server without an opt-in are all things this checks offline.

Runs start.sh's own PDF lines in a bash subshell with a stubbed `die`; reads scripts/config.sh, scripts/prepare.sh,
start.sh and README.md. No Docker, no GPU, no network.

Usage: tools/test_pdf_input.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = (ROOT / "scripts" / "config.sh").read_text(encoding="utf-8")
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


def one(text: str, pattern: str, name: str) -> str:
    hits = re.findall(pattern, text, re.M)
    if len(hits) != 1:
        return "" if not hits else hits[0]
    return hits[0]


# ---- the pin, and the fact that the image hash covers it -----------------------------------

version = one(CONFIG, r"^PYPDFIUM2_VERSION=(\S+)$", "version")
sha = one(CONFIG, r"^PYPDFIUM2_AARCH64_SHA256=([0-9a-f]+)$", "sha")
url = one(CONFIG, r'^PYPDFIUM2_WHEEL_URL="([^"]+)"', "url")
check(
    "config.sh pins the pypdfium2 wheel (version, URL, SHA-256)",
    ([] if re.fullmatch(r"\d+\.\d+\.\d+", version) else [f"PYPDFIUM2_VERSION={version!r}"])
    + ([] if re.fullmatch(r"[0-9a-f]{64}", sha) else [f"PYPDFIUM2_AARCH64_SHA256={sha!r} is not 64 hex digits"])
    + ([] if url.startswith("https://") and version and version in url else [f"URL {url!r} does not name version {version!r}"])
    + ([] if "aarch64" in url else [f"URL {url!r} is not the aarch64 wheel"]),
)
check(
    "image_hash() covers the PDFium pin, so a new pin builds a new image",
    [] if "pdfium-$PYPDFIUM2_VERSION-$PYPDFIUM2_AARCH64_SHA256" in CONFIG else
    ["the echo line of image_hash() does not include the PDFium version and SHA-256"],
)

# ---- the build: download, verify, extract, install, point the engine at it ------------------

dockerfile = PREPARE[PREPARE.find("<<'DOCKERFILE'"):]
check(
    "prepare.sh passes the pin into the build",
    [] if "--build-arg PYPDFIUM2_WHEEL_URL=" in PREPARE and "--build-arg PYPDFIUM2_AARCH64_SHA256=" in PREPARE
    else ["the docker build does not pass PYPDFIUM2_WHEEL_URL / PYPDFIUM2_AARCH64_SHA256"],
)
check(
    "the image checks the wheel's SHA-256 before extracting from it",
    [] if re.search(r'echo "\$\{PYPDFIUM2_AARCH64_SHA256\}\s+pypdfium2\.whl" \| sha256sum -c -.*?zipfile\.ZipFile\("pypdfium2\.whl"\)', dockerfile, re.S)
    else ["no sha256sum -c of the wheel ahead of the extraction (or none at all)"],
)
check(
    "the image installs libpdfium.so, names it for the engine and ships its licenses",
    ([] if 'z.read("pypdfium2_raw/libpdfium.so")' in dockerfile else ["does not extract pypdfium2_raw/libpdfium.so from the wheel"])
    + ([] if re.search(r"COPY --from=codecs .*libpdfium\.so /opt/tensorfold/lib/libpdfium\.so", dockerfile) else ["does not COPY libpdfium.so into /opt/tensorfold/lib"])
    + ([] if re.search(r"COPY --from=codecs /opt/codecs/pdfium/licenses /opt/tensorfold/share/doc/tensorfold/LICENSES-pdfium", dockerfile) else
       ["does not ship the wheel's license tree into the image docs"])
    + ([] if re.search(r"test -s /opt/codecs/pdfium/licenses/LICENSES/BSD-3-Clause\.txt", dockerfile) else
       ["does not check the extracted licenses are there"])
    + ([] if re.search(r"ENV [^#]*?TENSORFOLD_LIBPDFIUM=/opt/tensorfold/lib/libpdfium\.so", dockerfile, re.S)
       else ["TENSORFOLD_LIBPDFIUM is not set in the image ENV"]),
)

# ---- start.sh: --vision-pdf only when the operator asks for it ------------------------------

pdf_lines = [ln for ln in START.splitlines() if re.search(r'\[\[ "\$PDF"', ln)]
validator = [ln for ln in START.splitlines() if ln.startswith("for v in THINKING DRAFTS VISION VISION_URLS PDF")]
check("start.sh validates PDF and gates --vision-pdf on it",
      [] if len(pdf_lines) >= 2 and len(validator) == 1 else [f"found {len(validator)} validator line(s) and {len(pdf_lines)} PDF line(s)"])


def serve_args(pdf: str, vision: str) -> tuple[int, str]:
    """start.sh's own PDF lines with the settings it validates set; returns (exit status, the resulting SERVE_ARGS)."""
    script = ("die() { printf 'die: %s\\n' \"$*\" >&2; exit 3; }\n"
              "SERVE_ARGS=()\n"
              f"THINKING=1 DRAFTS=1 VISION_URLS=0 VISION={vision} PDF={pdf}\n"
              + "\n".join(validator + pdf_lines)
              + '\nprintf "%s\\n" "${SERVE_ARGS[*]}"\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


if pdf_lines and validator:
    rc, out = serve_args("1", "1")
    check("PDF=1 passes --vision-pdf", [] if rc == 0 and "--vision-pdf" in out else [f"rc={rc} args={out!r}"])
    rc, out = serve_args("0", "1")
    check("PDF=0 (the default) passes no --vision-pdf", [] if rc == 0 and "--vision-pdf" not in out else [f"rc={rc} args={out!r}"])
    rc, out = serve_args("2", "1")
    check("PDF=2 is refused", [] if rc == 3 else [f"rc={rc} args={out!r}, expected a refusal"])
    rc, out = serve_args("1", "0")
    check("PDF=1 without VISION=1 is refused", [] if rc == 3 else [f"rc={rc} args={out!r}, expected a refusal"])

# ---- README says what it does and what the engine says when it is off ------------------------

check(
    "README documents the PDF opt-in",
    ([] if "PDF=1 ./start.sh restart" in README else ["no `PDF=1 ./start.sh restart` instruction"])
    + ([] if "--vision-pdf" in README else ["the README never names --vision-pdf"])
    + ([] if "PDF inputs are off on this server" in README else ["the README does not quote the engine's refusal when the opt-in is off"])
    + ([] if re.search(r"^\| `PDF` \| `0` \|", README, re.M) else ["the Configuration table has no PDF row"]),
)

if failures:
    print(f"\n{len(failures)} PDF check(s) failed")
    sys.exit(1)
print("\nall passed")
