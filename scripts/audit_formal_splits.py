"""Assign deterministic train/dev partitions and audit the formal manifest."""

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


INPUT = Path("datasets/manifests/formal_sft.jsonl")
OUTPUT = Path("datasets/manifests/formal_sft_split.jsonl")
INDEX = Path("datasets/indices/formal_search/evidence.sqlite3")


def load_rows():
    return [json.loads(line) for line in INPUT.read_text().splitlines()]


def assign_partitions(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["group_id"])].append(row)
    ordered = sorted(groups)
    partitions = {}
    for position, key in enumerate(ordered):
        partitions[key] = "sft_dev" if position % 10 == 0 else "sft_train"
    for row in rows:
        row["partition"] = partitions[(row["dataset"], row["group_id"])]


def audit(rows):
    ids = [row["sample_id"] for row in rows]
    groups = [(row["dataset"], row["group_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate sample_id")
    if len(groups) != len(set(groups)):
        raise RuntimeError("duplicate dataset/group_id")
    allowed_actions = {"CROP", "SEARCH", "ANSWER"}
    connection = sqlite3.connect(INDEX)
    missing_images = []
    missing_evidence = []
    invalid_actions = []
    for row in rows:
        if not Path(row["image_path"]).exists():
            missing_images.append(row["sample_id"])
        for step in row["trajectory"]:
            if step["action"] not in allowed_actions:
                invalid_actions.append(row["sample_id"])
        for evidence_id in row["search_evidence_ids"]:
            found = connection.execute(
                "SELECT 1 FROM evidence WHERE evidence_id = ?", (evidence_id,)
            ).fetchone()
            if found is None:
                missing_evidence.append(evidence_id)
    connection.close()
    if missing_images or missing_evidence or invalid_actions:
        raise RuntimeError(
            f"missing_images={len(missing_images)} missing_evidence={len(missing_evidence)} "
            f"invalid_actions={len(invalid_actions)}"
        )


def main():
    rows = load_rows()
    assign_partitions(rows)
    audit(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter(f"{row['dataset']}:{row['partition']}" for row in rows)
    print(json.dumps({"rows": len(rows), "partitions": counts}))


if __name__ == "__main__":
    main()
