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
    parser.add_argument("--teacher-glob", default="artifacts/sft-full/teacher/*.jsonl")
    parser.add_argument("--recovery-glob", default="artifacts/sft-full/recovery/*.jsonl")
    parser.add_argument("--output", default="artifacts/sft-full/formal_sft.jsonl")
    parser.add_argument("--oracle-count", type=int, default=3600)
    parser.add_argument("--teacher-count", type=int, default=1500)
    parser.add_argument("--recovery-count", type=int, default=900)
    args = parser.parse_args()

    oracle = [json.loads(line) for line in Path(args.oracle).read_text().splitlines()]
    oracle = [record for record in oracle if record["partition"] == "sft_train"][:args.oracle_count]
    teacher = read_jsonl(sorted(glob.glob(args.teacher_glob)))
    recovery = read_jsonl(sorted(glob.glob(args.recovery_glob)))
    teacher = [record for record in teacher if record["trajectory_type"] == "teacher"][:args.teacher_count]
    recovery = [record for record in recovery if record["trajectory_type"] == "recovery"][:args.recovery_count]
    if len(oracle) != args.oracle_count or len(teacher) != args.teacher_count or len(recovery) != args.recovery_count:
        raise RuntimeError(
            f"insufficient SFT mixture: oracle={len(oracle)}, teacher={len(teacher)}, recovery={len(recovery)}"
        )
    records = oracle + teacher + recovery
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "count": len(records),
        "oracle": len(oracle),
        "teacher": len(teacher),
        "recovery": len(recovery),
    }))


if __name__ == "__main__":
    main()
