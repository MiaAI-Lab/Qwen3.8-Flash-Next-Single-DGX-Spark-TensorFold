#!/usr/bin/env python3
"""Video parts are opt-in, and the image builds libtfvideo over a checksummed FFmpeg (issue #90, VIDEO=1).

The CUDA engine takes a video only when it can dlopen `libtfvideo.so`, which the recipe builds over the FFmpeg release
TensorFold's tools/zig/build_ffmpeg.sh pins. So this checks, offline: that the FFmpeg pin is a pin and is part of the image
hash, that the build verifies the tarball before untarring it and keeps the library off the paths the server probes (VIDEO=0
must not link FFmpeg at all), that start.sh only reaches for it with VIDEO=1, and that the README says what the wiring does.

Runs start.sh's own VIDEO lines in a bash subshell with a stubbed `die`; reads scripts/config.sh, scripts/prepare.sh,
start.sh, README.md and patches/. No Docker, no GPU, no network.

Usage: tools/test_video_input.py
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


def one(text: str, pattern: str) -> str:
    hits = re.findall(pattern, text, re.M)
    return hits[0] if hits else ""


# ---- the patches stay one numbered series, with the video one in it ---------------------------

numbers = sorted(int(p.name[:4]) for p in (ROOT / "patches").glob("[0-9][0-9][0-9][0-9]-*.patch"))
check(
    "patches/ is one unbroken series that includes the video patch",
    ([] if numbers == list(range(1, len(numbers) + 1)) else [f"patch numbers are {numbers}, expected 1..{len(numbers)}"])
    + ([] if list((ROOT / "patches").glob("0005-*video*.patch")) else ["no 0005 video patch in patches/"]),
)

# ---- the pin, and the fact that the image hash covers it -------------------------------------

version = one(CONFIG, r"^FFMPEG_VERSION=(\S+)$")
sha = one(CONFIG, r"^FFMPEG_SHA256=([0-9a-f]+)$")
check(
    "config.sh pins the FFmpeg release by version and SHA-256",
    ([] if re.fullmatch(r"\d+\.\d+\.\d+", version) else [f"FFMPEG_VERSION={version!r}"])
    + ([] if re.fullmatch(r"[0-9a-f]{64}", sha) else [f"FFMPEG_SHA256={sha!r} is not 64 hex digits"]),
)
check(
    "image_hash() covers the FFmpeg pin, so a new release builds a new image",
    [] if "ffmpeg-$FFMPEG_VERSION-$FFMPEG_SHA256" in CONFIG else
    ["the echo line of image_hash() does not include the FFmpeg version and SHA-256"],
)

# ---- the build: verify the tarball, build it, keep the library where the server cannot see it --

dockerfile = PREPARE[PREPARE.find("<<'DOCKERFILE'"):]
check(
    "prepare.sh passes the pin into the build",
    [] if "--build-arg FFMPEG_VERSION=" in PREPARE and "--build-arg FFMPEG_SHA256=" in PREPARE
    else ["the docker build does not pass FFMPEG_VERSION / FFMPEG_SHA256"],
)
check(
    "the image checks the tarball's SHA-256 before untarring it, and hands the source to TensorFold's own script",
    [] if re.search(r'echo "\$\{FFMPEG_SHA256\}\s+ffmpeg\.tar\.xz" \| sha256sum -c - && tar -xJf ffmpeg\.tar\.xz && \\\s*\n\s*/tensorfold/tools/zig/build_ffmpeg\.sh /opt/ffmpeg /tmp/ffmpeg', dockerfile)
    else ["no sha256sum -c ahead of the tar, or tools/zig/build_ffmpeg.sh is not run with the source dir"],
)
build = one(dockerfile, r"RUN cd /tensorfold && /opt/zig/zig build[^#]*?\n(?:.*\\\n)*.*\n")
check(
    "the image builds libtfvideo with `zig build video` over that FFmpeg, and then hides it",
    ([] if " fatbins native video " in build else ["the zig build does not ask for the `video` target"])
    + ([] if "-Dffmpeg=/opt/ffmpeg" in build else ["the zig build does not point -Dffmpeg at the FFmpeg we built"])
    + ([] if re.search(r"test -s /opt/tensorfold/native/lib/libtfvideo\.so", build) else ["the build does not check libtfvideo.so was produced"])
    + ([] if re.search(r"mv /opt/tensorfold/native/lib/libtfvideo\.so /opt/tensorfold/libexec/tfvideo/libtfvideo\.so", build) else
       ["libtfvideo.so stays in native/lib, where the server's dlopen would find it with VIDEO=0"])
    + ([] if re.search(r"test ! -e /opt/tensorfold/native/lib/libtfvideo\.so", build) else
       ["nothing proves native/lib lost the library after the move"]),
)
check(
    "the image carries FFmpeg's license and the archives it was linked from",
    ([] if re.search(r'cp "ffmpeg-\$\{FFMPEG_VERSION\}/COPYING\.LGPLv2\.1" "ffmpeg-\$\{FFMPEG_VERSION\}/LICENSE\.md"', dockerfile) else ["FFmpeg's LGPL-2.1 license is not copied"])
    + ([] if "cp /opt/ffmpeg/lib/*.a /opt/ffmpeg/share/doc/ffmpeg/objects/" in dockerfile else ["the static archives FFmpeg was linked from are not kept (LGPL-2.1 section 6)"])
    + ([] if re.search(r"cp -a /opt/ffmpeg/share/doc/ffmpeg /opt/tensorfold/share/doc/ffmpeg", dockerfile) else ["they are not copied into the image's docs"]),
)

# ---- start.sh: the library is reachable, and video is asked for, only with VIDEO=1 -------------

video_lines = [ln for ln in START.splitlines() if ln.lstrip().startswith('[[ "$VIDEO"')]
validator = [ln for ln in START.splitlines() if ln.startswith("for v in THINKING DRAFTS VISION VISION_URLS PDF VIDEO")]
tokens = [ln for ln in START.splitlines() if "TENSORFOLD_VIDEO_TOKENS >= 64" in ln]
# a check split over two lines would merge with the next one when run on its own, so the extracted lines must each stand alone
dangling = [ln for ln in validator + tokens + video_lines if ln.rstrip().endswith("||") or ln.rstrip().endswith("&&")]
check("start.sh validates VIDEO, gates the decoder path and the video flags on it",
      [] if len(validator) == 1 and len(video_lines) >= 3 and len(tokens) == 1 and not dangling else
      [f"found {len(validator)} validator line(s), {len(video_lines)} VIDEO line(s), {len(tokens)} token-range line(s)"
       + (f" and {len(dangling)} that run on past the line end: {dangling}" if dangling else "")])


def serve_args(video: str, vision: str, tokens_value: str = "16384") -> tuple[int, str, str]:
    """start.sh's own VIDEO lines with the settings it validates set; returns (exit status, SERVE_ARGS, CMD_PREFIX)."""
    names = re.findall(r"for v in ([A-Z_ ]+);", validator[0])[0].split()
    env = " ".join(f"{n}={dict(VIDEO=video, VISION=vision).get(n, '1')}" for n in names)
    script = ("die() { printf 'die: %s\\n' \"$*\" >&2; exit 3; }\n"
              "SERVE_ARGS=()\n"
              "CMD_PREFIX=()\n"
              f"TENSORFOLD_VIDEO_TOKENS={tokens_value}\n"
              f"{env}\n"
              + "\n".join(validator + tokens + video_lines)
              + '\nprintf "%s|%s\\n" "${SERVE_ARGS[*]}" "${CMD_PREFIX[*]}"\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    serve, _, prefix = r.stdout.strip().partition("|")
    return r.returncode, serve, prefix


if validator and video_lines and tokens:
    rc, serve, prefix = serve_args("1", "1")
    check("VIDEO=1 passes --vision-video-tokens and puts libtfvideo on the loader's path",
          [] if rc == 0 and "--vision-video-tokens 16384" in serve and "/opt/tensorfold/libexec/tfvideo" in prefix
          else [f"rc={rc} args={serve!r} prefix={prefix!r}"])
    rc, serve, prefix = serve_args("0", "1")
    check("VIDEO=0 (the default) passes no video flag and leaves the decoder off the path",
          [] if rc == 0 and "--vision-video-tokens" not in serve and prefix == "" else [f"rc={rc} args={serve!r} prefix={prefix!r}"])
    rc, _, _ = serve_args("2", "1")
    check("VIDEO=2 is refused", [] if rc == 3 else [f"rc={rc}, expected a refusal"])
    rc, _, _ = serve_args("1", "0")
    check("VIDEO=1 without VISION=1 is refused", [] if rc == 3 else [f"rc={rc}, expected a refusal"])
    rc, serve, _ = serve_args("1", "1", "40")
    check("a video token budget outside the engine's range is refused", [] if rc == 3 else [f"rc={rc} args={serve!r}, expected a refusal"])

# ---- README says what it does, and what is still unmeasured ------------------------------------

check(
    "README documents the video opt-in",
    ([] if "VIDEO=1 ./start.sh restart" in README else ["no `VIDEO=1 ./start.sh restart` instruction"])
    + ([] if "--vision-video-tokens" in README else ["the README never names --vision-video-tokens"])
    + ([] if "libtfvideo" in README else ["the README never names libtfvideo"])
    + ([] if "/opt/tensorfold/libexec/tfvideo" in README else ["the README does not say the library sits off the probed paths"])
    + ([] if re.search(r"^\| `VIDEO` \| `0` \|", README, re.M) else ["the Configuration table has no VIDEO row"])
    + ([] if "compiles" in README and "Not measured on a Spark yet" in README else ["the README does not say the video path is unmeasured on a Spark"]),
)

if failures:
    print(f"\n{len(failures)} video check(s) failed")
    sys.exit(1)
print("\nall passed")
