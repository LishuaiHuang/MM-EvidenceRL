#!/usr/bin/env python
"""Assemble the Oracle, Teacher, and Recovery SFT mixture."""

import argparse
import glob
import json
from pathlib import Path


def read_jsonl(paths):
    records = []
    for path in paths:
        records.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default="datasets/manifests/formal_sft_split.jsonl")
    parser.add_argument("--teacher-glob", default="artifacts/sft-full/teacher-canonical/part-*.jsonl")
    parser.add_argument("--recovery-glob", default="artifacts/sft-full/recovery/*.jsonl")
    parser.add_argument("--output", default="artifacts/sft-full/formal_sft.jsonl")
    parser.add_argument("--target-count", type=int, default=5400)
    parser.add_argument("--oracle-count", type=int)
    parser.add_argument("--teacher-count", type=int)
    parser.add_argument("--recovery-count", type=int)
    args = parser.parse_args()

    oracle = [json.loads(line) for line in Path(args.oracle).read_text().splitlines()]
    oracle = [record for record in oracle if record["partition"] == "sft_train"]
    teacher = [record for record in read_jsonl(sorted(glob.glob(args.teacher_glob)))
               if record["trajectory_type"] == "teacher" and "trajectory" in record]
    recovery = [record for record in read_jsonl(sorted(glob.glob(args.recovery_glob)))
                if record["trajectory_type"] == "recovery" and "trajectory" in record]
    if args.teacher_count is not None:
        teacher = teacher[:args.teacher_count]
    if args.recovery_count is not None:
        recovery = recovery[:args.recovery_count]

    selected = []
    seen = set()
    for record in teacher + recovery:
        if record["sample_id"] in seen:
            continue
        selected.append(record)
        seen.add(record["sample_id"])
    oracle = [record for record in oracle if record["sample_id"] not in seen]
    if args.oracle_count is not None:
        oracle = oracle[:args.oracle_count]
    selected.extend(oracle[:max(0, args.target_count - len(selected))])
    if len(selected) < args.target_count:
        raise RuntimeError(f"insufficient unique SFT records: available={len(selected)}, target={args.target_count}")
    records = selected[:args.target_count]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "count": len(records),
        "oracle": sum(record.get("trajectory_type") not in {"teacher", "recovery"} for record in records),
        "teacher": sum(record.get("trajectory_type") == "teacher" for record in records),
        "recovery": sum(record.get("trajectory_type") == "recovery" for record in records),
    }))


if __name__ == "__main__":
    main()
