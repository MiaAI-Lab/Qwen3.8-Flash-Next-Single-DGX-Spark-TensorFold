#!/usr/bin/env python3
"""stop.sh does not run what /health says (issue #82).

stop.sh reads the number of running requests from the server's /health and compares it in `(( busy == 0 ))`, where bash
evaluates the value as arithmetic: an array subscript on a variable that exists, `HOME[$(cmd)]`, runs `cmd` on the host
(a name that does not exist stops stop.sh under `set -u`, before it stops the container). This runs the
real stop.sh with a `docker` and a `curl` on PATH that report a running container and a /health answer whose `running`
is such a string, and checks that the command did not run, that the number of requests is read when it is a number, and that stop.sh
still stops the container either way.

Usage: tools/test_stop_health.py        (needs bash and python3; no GPU, no Docker, no server)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCKER = """#!/bin/sh
case "$1" in
  ps) echo qwen38-flash-next-tf ;;
  inspect) echo true ;;
esac
echo "docker $*" >> "$FAKE_LOG"
"""
CURL = """#!/bin/sh
printf '%s' "$FAKE_HEALTH"
"""


def run_stop(tmp: Path, health: str) -> tuple[subprocess.CompletedProcess, str]:
    log = tmp / "docker.log"
    log.write_text("")
    env = {**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}", "LC_ALL": "C", "FAKE_HEALTH": health, "FAKE_LOG": str(log),
           "STATE_DIR": str(tmp / "state"), "HOME": str(tmp)}
    r = subprocess.run(["bash", str(ROOT / "stop.sh")], capture_output=True, text=True, env=env, cwd=tmp)
    return r, log.read_text()


def main() -> int:
    fails = []

    def check(ok: bool, what: str) -> None:
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        for name, body in (("docker", DOCKER), ("curl", CURL)):
            (tmp / name).write_text(body)
            (tmp / name).chmod(0o755)
        marker = tmp / "ran"

        for what, health in (("a string subscript", '{"running": "HOME[$(touch %s)]"}' % marker),
                             ("a string subscript in requests_running", '{"requests_running": "HOME[$(touch %s)]"}' % marker),
                             ("an expression", '{"running": "1+HOME[$(touch %s)]"}' % marker),
                             ("a name that is not set", '{"running": "nothing"}')):
            r, log = run_stop(tmp, health)
            check(not marker.exists(), f"{what} in /health does not run on the host")
            marker.unlink(missing_ok=True)
            check(r.returncode == 0 and "docker stop" in log and "docker rm -f" in log, f"stop.sh still stops the container ({what})")
            check("still running" not in r.stderr + r.stdout, f"{what} counts as no running requests")

        for health, what in (('{"running": true}', "a boolean"), ('{"running": -3}', "a negative number"), ("not json", "no JSON"), ("", "no answer")):
            r, log = run_stop(tmp, health)
            check(r.returncode == 0 and "docker stop" in log and "still running" not in r.stderr + r.stdout, f"{what}: stops, counts 0")

        r, log = run_stop(tmp, '{"running": 3}')
        check(r.returncode == 0 and "3 request(s) still running will be cut off" in r.stderr + r.stdout, "a number of running requests is reported")
        r, log = run_stop(tmp, '{"requests_running": 2}')
        check("2 request(s) still running" in r.stderr + r.stdout, "requests_running is read when running is absent")

    print(f"{len(fails)} failed" if fails else "all passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
