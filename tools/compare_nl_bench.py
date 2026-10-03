#!/usr/bin/env python3
"""Validate paired A/B records and report speed, latency and exact-output evidence.

Nonzero exit for failed/cached/incomplete/mismatched pairs. No improvement threshold
is asserted: the report includes regressions and separates Dutch from controls.
"""
import argparse
import json
from pathlib import Path
import statistics
from collections import defaultdict


def key(row):
    return (row["block"], row["clients"], row["repeat"], row["name"])


def load(paths):
    rows = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            ident = key(row)
            if ident in rows:
                raise ValueError(f"duplicate pairing key {ident}")
            rows[ident] = row
    if not rows:
        raise ValueError("no benchmark records")
    return rows


def compare(a, b):
    if a.keys() != b.keys():
        raise ValueError("A/B pairing keys differ; no dropping failed/missing runs")
    groups, issues = defaultdict(list), []
    for ident, x in a.items():
        y = b[ident]
        if x.get("arm") != "A" or y.get("arm") != "B":
            issues.append(f"{ident}: wrong arm labels")
        for field in ("manifest_sha256", "request_sha256", "group", "temperature", "thinking", "seed", "max_tokens", "actual_clients"):
            if x.get(field) is None or x.get(field) != y.get(field):
                issues.append(f"{ident}: {field} differs or is absent")
        for arm, row in (("A", x), ("B", y)):
            if not row.get("ok") or row.get("error") or row.get("finish_reason") not in ("stop", "length"):
                issues.append(f"{ident} {arm}: failed or incomplete request")
            if row.get("cached") is None or row["cached"] != 0:
                issues.append(f"{ident} {arm}: missing cache count or cached prompt")
            for field in ("decode_s", "completion_tokens", "ttft", "elapsed"):
                if not isinstance(row.get(field), (int, float)) or row[field] <= 0:
                    issues.append(f"{ident} {arm}: invalid {field}")
        # Exact IDs (including reasoning if exposed), then the engine token digest.
        exact = None
        if x.get("token_ids") and y.get("token_ids"):
            exact = x["token_ids"] == y["token_ids"]
        elif x.get("token_sha") and y.get("token_sha"):
            exact = x["token_sha"] == y["token_sha"]
        if exact is not True or not x.get("output_sha") or x["output_sha"] != y.get("output_sha") or x.get("completion_tokens") != y.get("completion_tokens") or x.get("finish_reason") != y.get("finish_reason"):
            issues.append(f"{ident}: exact token/output/length equality failed or unavailable")
        if all(isinstance(r.get("decode_s"), (int, float)) and r["decode_s"] > 0 and
               isinstance(r.get("completion_tokens"), (int, float)) and r["completion_tokens"] > 0 for r in (x, y)):
            groups[(x["group"], x["clients"])].append((x, y))
    report = {"valid": not issues, "pairs": len(a), "issues": issues, "groups": []}
    for (group, clients), pairs in sorted(groups.items()):
        speed_a = [x["completion_tokens"] / x["decode_s"] for x, _ in pairs]
        speed_b = [y["completion_tokens"] / y["decode_s"] for _, y in pairs]
        entry = {"group": group, "clients": clients, "pairs": len(pairs),
                 "A_median_decode_tps": statistics.median(speed_a),
                 "B_median_decode_tps": statistics.median(speed_b),
                 "median_paired_speed_ratio": statistics.median(v/u for u, v in zip(speed_a, speed_b))}
        for arm, index in (("A", 0), ("B", 1)):
            rows = [pair[index] for pair in pairs]
            for field in ("ttft", "elapsed", "batch_wall_s"):
                values = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
                entry[f"{arm}_median_{field}"] = statistics.median(values) if values else None
            drafted = sum(r.get("drafted") or 0 for r in rows)
            accepted = sum(r.get("accepted") or 0 for r in rows)
            entry[f"{arm}_acceptance"] = accepted/drafted if drafted else None
            entry[f"{arm}_missing_draft_stats"] = sum(r.get("drafted") is None or r.get("accepted") is None for r in rows)
        report["groups"].append(entry)
    return report


def serial_equality(drafted, serial):
    if drafted.keys() != serial.keys():
        return ["serial pairing keys differ"]
    issues = []
    for ident, row in drafted.items():
        ref = serial[ident]
        if ref.get("arm") != "serial" or ref.get("draft") is not False or not ref.get("ok") or ref.get("error"):
            issues.append(f"{ident}: invalid serial control")
        for field in ("request_sha256", "manifest_sha256", "temperature", "thinking", "seed", "max_tokens"):
            if row.get(field) is None or row.get(field) != ref.get(field):
                issues.append(f"{ident}: serial {field} mismatch")
        equal_ids = (row["token_ids"] == ref["token_ids"] if row.get("token_ids") and ref.get("token_ids")
                     else bool(row.get("token_sha") and row["token_sha"] == ref.get("token_sha")))
        if not equal_ids or not row.get("output_sha") or row["output_sha"] != ref.get("output_sha") or row.get("completion_tokens") != ref.get("completion_tokens") or row.get("finish_reason") != ref.get("finish_reason"):
            issues.append(f"{ident}: serial exact output equality failed or unavailable")
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--serial-a", nargs="+", help="serial control files from the A image")
    ap.add_argument("--serial-b", nargs="+", help="serial control files from the B image")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    report = compare(a, b)
    report["serial_checked"] = {"A": bool(args.serial_a), "B": bool(args.serial_b)}
    for name, rows, paths in (("A", a, args.serial_a), ("B", b, args.serial_b)):
        if paths:
            report["issues"].extend(f"{name}: {issue}" for issue in serial_equality(rows, load(paths)))
    report["valid"] = not report["issues"]
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
