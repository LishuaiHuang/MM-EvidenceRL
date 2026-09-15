"""Build the small, replayable E0 manifest from downloaded source files."""

import json
from pathlib import Path

import pyarrow.parquet as parquet


CHARTQA_REV = "af8b6f5c08c95085271561c2a3f9d15f2b5a9031"
FVQA_REV = "bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5"


def chartqa_records(root, limit=20):
    split_dir = root / "ChartQA Dataset" / "train"
    questions = json.loads((split_dir / "train_human.json").read_text())
    records = []
    for item in questions:
        image_name = item["imgname"]
        image = split_dir / "png" / image_name
        annotation = split_dir / "annotations" / f"{Path(image_name).stem}.json"
        if not image.exists() or not annotation.exists():
            continue
        ann = json.loads(annotation.read_text())
        figure = ann.get("general_figure_info", {}).get("figure_info", {})
        bbox = figure.get("bbox")
        if not bbox:
            continue
        width = bbox["w"]
        height = bbox["h"]
        records.append(
            {
                "sample_id": f"chartqa_{Path(image_name).stem}_{len(records):04d}",
                "dataset": "chartqa",
                "dataset_revision": CHARTQA_REV,
                "split": "train",
                "group_id": Path(image_name).stem,
                "task_mode": "perception",
                "image_path": str(image),
                "question": item["query"],
                "answer": item["label"],
                "answer_type": "text",
                "region_type": "figure",
                "gold_bboxes_root": [[
                    bbox["x"], bbox["y"], bbox["x"] + width, bbox["y"] + height
                ]],
                "requires_search": False,
                "search_evidence_ids": [],
                "trajectory": [
                    {"action": "CROP", "bbox_index": 0},
                    {"action": "ANSWER", "answer": item["label"]},
                ],
            }
        )
        if len(records) == limit:
            break
    return records


def fvqa_records(path, output_dir, search_count=15, free_count=5):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    counts = {"search_required": 0, "search_free": 0}
    parquet_file = parquet.ParquetFile(path)
    columns = ["prompt", "reward_model", "data_id", "category", "images"]
    for batch in parquet_file.iter_batches(columns=columns, batch_size=32):
        for row in batch.to_pylist():
            category = row["category"]
            if category not in counts:
                continue
            if category == "search_required" and counts[category] >= search_count:
                continue
            if category == "search_free" and counts[category] >= free_count:
                continue
            image_bytes = row["images"][0]["bytes"]
            sample_id = row["data_id"]
            image_path = output_dir / f"{sample_id}.png"
            image_path.write_bytes(image_bytes)
            question = row["prompt"][0]["content"]
            answer = row["reward_model"]["ground_truth"]
            evidence_id = f"{sample_id}_oracle"
            requires_search = category == "search_required"
            records.append(
                {
                    "sample_id": sample_id,
                    "dataset": "fvqa",
                    "dataset_revision": FVQA_REV,
                    "split": "train",
                    "group_id": sample_id,
                    "task_mode": "knowledge",
                    "image_path": str(image_path),
                    "question": question,
                    "answer": answer,
                    "answer_type": "text",
                    "region_type": "root",
                    "gold_bboxes_root": [],
                    "requires_search": requires_search,
                    "search_evidence_ids": [evidence_id] if requires_search else [],
                    "trajectory": [
                        {"action": "SEARCH", "query": question}
                        if requires_search
                        else {"action": "ANSWER", "answer": answer},
                        {"action": "ANSWER", "answer": answer},
                    ],
                }
            )
            counts[category] += 1
            if counts["search_required"] == search_count and counts["search_free"] == free_count:
                return records
    return records


def main():
    chartqa_root = Path("datasets/raw/chartqa/extracted")
    fvqa_path = Path("datasets/raw/fvqa/fvqa_train.parquet")
    e0_root = Path("datasets/e0")
    perception = chartqa_records(chartqa_root)
    knowledge = fvqa_records(fvqa_path, e0_root / "images")
    records = perception + knowledge
    evidence = [
        {
            "evidence_id": record["search_evidence_ids"][0],
            "sample_id": record["sample_id"],
            "source": "fvqa_ground_truth_oracle",
            "text": record["answer"],
        }
        for record in knowledge
        if record["requires_search"]
    ]
    manifest_dir = Path("datasets/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "e0.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (e0_root / "evidence.jsonl").open("w") as handle:
        for item in evidence:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(perception)} perception, {len(knowledge)} knowledge, {len(evidence)} evidence")


if __name__ == "__main__":
    main()
