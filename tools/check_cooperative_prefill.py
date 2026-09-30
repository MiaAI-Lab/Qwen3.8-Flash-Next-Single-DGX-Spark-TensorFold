#!/usr/bin/env python3
"""Apply the pinned recipe to a temporary copy of an existing TensorFold checkout.

No download, Docker, model import or GPU. Requires git, patch, Python and NumPy.
The optional language patch is checked separately, in Docker's basename order.
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "v0.3.6.3"
COMMIT = "191188075bca56a7c71074a79375eb4c1cb22e1c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="existing TensorFold git checkout with v0.3.6.3")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", TAG + "^{commit}"], text=True
    ).strip()
    if revision != COMMIT:
        raise SystemExit("v0.3.6.3 does not resolve to the expected commit")
    archive = subprocess.check_output(["git", "-C", str(args.upstream), "archive", COMMIT, "src"])
    regular = sorted((ROOT / "patches").glob("*.patch"))
    language = list((ROOT / "patches" / "languages").glob("*.patch"))
    with tempfile.TemporaryDirectory(prefix="tensorfold-prefill-") as tmp:
        tmp = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(tmp, filter="data")
        for label, patches in (("default", regular), ("languages", regular + language)):
            dest = tmp / label
            shutil.copytree(tmp / "src", dest)
            for patch in sorted(patches, key=lambda p: p.name):
                print(f"{label}: {patch.name}", flush=True)
                with patch.open("rb") as source:
                    subprocess.run(
                        ["patch", "--batch", "--forward", "--fuzz=0", "-p0"], cwd=dest, stdin=source, check=True
                    )
            env = dict(os.environ, TENSORFOLD_SOURCE=str(dest / "tensorfold"))
            for suite in ("test_cooperative_prefill.py", "test_cooperative_prefill_state.py"):
                subprocess.run([sys.executable, "-B", str(ROOT / "tests" / suite)], env=env, check=True)
    print("Both patch chains and offline suites passed; no GPU/container validation performed.")


if __name__ == "__main__":
    main()
