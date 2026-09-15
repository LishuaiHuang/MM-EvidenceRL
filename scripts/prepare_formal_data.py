"""Prepare the first 6,000 protocol-ready Oracle trajectories and local evidence index."""

import json
import pickle
import sqlite3
from pathlib import Path

import pyarrow.parquet as parquet
from PIL import Image


CHARTQA_REV = "af8b6f5c08c95085271561c2a3f9d15f2b5a9031"
FVQA_REV = "bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5"
CHARTQA_LIMIT = 4200
FVQA_LIMIT = 1800


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def build_evidence_index(cache_path, index_path):
    Image.Image.__setstate__ = lambda self, state: None
    cache = pickle.loads(cache_path.read_bytes())
    index_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_path)
    connection.executescript(
        """
        DROP TABLE IF EXISTS evidence;
        DROP TABLE IF EXISTS evidence_fts;
        CREATE TABLE evidence (
            evidence_id TEXT PRIMARY KEY,
            sample_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            title TEXT NOT NULL,
            url TEXT,
            source TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE evidence_fts USING fts5(
            evidence_id UNINDEXED, sample_id UNINDEXED, title, url
        );
        CREATE INDEX evidence_sample_rank ON evidence(sample_id, rank);
        """
    )
    rows = []
    for sample_id, result in cache.items():
        titles = result.get("tool_returned_web_title_list", [])
        urls = result.get("tool_returned_images_urls", [])
        for rank, title in enumerate(titles, 1):
            evidence_id = f"{sample_id}:search:{rank}"
            url = urls[rank - 1] if rank <= len(urls) and isinstance(urls[rank - 1], str) else None
            rows.append((evidence_id, sample_id, rank, title, url, "fvqa_search_cache"))
    connection.executemany("INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?)", rows)
    connection.executemany(
        "INSERT INTO evidence_fts(evidence_id, sample_id, title, url) VALUES (?, ?, ?, ?)",
        [(row[0], row[1], row[3], row[4] or "") for row in rows],
    )
    connection.commit()
    connection.close()
    return cache


def chartqa_records(root, excluded_groups):
    split_dir = root / "ChartQA Dataset" / "train"
    records = []
    seen_groups = set(excluded_groups)
    for filename in ("train_human.json", "train_augmented.json"):
        for item in json.loads((split_dir / filename).read_text()):
            group_id = Path(item["imgname"]).stem
            if group_id in seen_groups:
                continue
            image = split_dir / "png" / item["imgname"]
            annotation = split_dir / "annotations" / f"{group_id}.json"
            if not image.exists() or not annotation.exists():
                continue
            figure = json.loads(annotation.read_text()).get("general_figure_info", {}).get("figure_info", {})
            bbox = figure.get("bbox")
            if not bbox:
                continue
            x0, y0 = bbox["x"], bbox["y"]
            x1, y1 = x0 + bbox["w"], y0 + bbox["h"]
            sample_id = f"chartqa_{group_id}_{len(records):05d}"
            crop_evidence_id = f"{sample_id}:crop:0"
            records.append(
                {
                    "sample_id": sample_id,
                    "dataset": "chartqa",
                    "dataset_revision": CHARTQA_REV,
                    "split": "train",
                    "partition": "sft_train",
                    "group_id": group_id,
                    "task_mode": "perception",
                    "trajectory_type": "oracle",
                    "image_path": str(image),
                    "question": item["query"],
                    "answer": item["label"],
                    "answer_type": "text",
                    "region_type": "figure",
                    "gold_bboxes_root": [[x0, y0, x1, y1]],
                    "requires_search": False,
                    "search_evidence_ids": [],
                    "answer_evidence_id": crop_evidence_id,
                    "trajectory": [
                        {"action": "CROP", "bbox": [x0, y0, x1, y1], "evidence_id": crop_evidence_id},
                        {"action": "ANSWER", "answer": item["label"], "evidence_id": crop_evidence_id},
                    ],
                }
            )
            seen_groups.add(group_id)
            if len(records) == CHARTQA_LIMIT:
                return records
    return records


def choose_evidence(sample_id, answer, cache):
    result = cache[sample_id]
    titles = result.get("tool_returned_web_title_list", [])
    answer_text = str(answer).lower()
    for rank, title in enumerate(titles, 1):
        if answer_text and answer_text in title.lower():
            return f"{sample_id}:search:{rank}"
    return f"{sample_id}:search:1"


def fvqa_records(path, image_dir, excluded_ids, cache):
    image_dir.mkdir(parents=True, exist_ok=True)
    targets = {"search_required": 1260, "search_free": 540}
    counts = {key: 0 for key in targets}
    records = []
    columns = ["prompt", "reward_model", "data_id", "category", "images"]
    for batch in parquet.ParquetFile(path).iter_batches(columns=columns, batch_size=32):
        for row in batch.to_pylist():
            sample_id = row["data_id"]
            category = row["category"]
            if sample_id in excluded_ids or category not in targets or counts[category] >= targets[category]:
                continue
            if category == "search_required" and sample_id not in cache:
                continue
            image_path = image_dir / f"{sample_id}.png"
            image_path.write_bytes(row["images"][0]["bytes"])
            question = row["prompt"][0]["content"]
            answer = row["reward_model"]["ground_truth"]
            requires_search = category == "search_required"
            evidence_id = choose_evidence(sample_id, answer, cache) if requires_search else f"{sample_id}:root_image"
            records.append(
                {
                    "sample_id": sample_id,
                    "dataset": "fvqa",
                    "dataset_revision": FVQA_REV,
                    "split": "train",
                    "partition": "sft_train",
                    "group_id": sample_id,
                    "task_mode": "knowledge",
                    "trajectory_type": "oracle",
                    "image_path": str(image_path),
                    "question": question,
                    "answer": answer,
                    "answer_type": "text",
                    "region_type": "root",
                    "gold_bboxes_root": [],
                    "requires_search": requires_search,
                    "search_evidence_ids": [evidence_id] if requires_search else [],
                    "answer_evidence_id": evidence_id,
                    "trajectory": [
                        *([{"action": "SEARCH", "query": question, "evidence_ids": [evidence_id]}] if requires_search else []),
                        {"action": "ANSWER", "answer": answer, "evidence_id": evidence_id},
                    ],
                }
            )
            counts[category] += 1
            if sum(counts.values()) == FVQA_LIMIT:
                return records
    return records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    e0 = read_jsonl(Path("datasets/manifests/e0.jsonl"))
    excluded_chart_groups = {r["group_id"] for r in e0 if r["dataset"] == "chartqa"}
    excluded_fvqa_ids = {r["sample_id"] for r in e0 if r["dataset"] == "fvqa"}
    cache = build_evidence_index(
        Path("datasets/raw/fvqa/fvqa_train_image_search_results_cache.pkl"),
        Path("datasets/indices/formal_search/evidence.sqlite3"),
    )
    perception = chartqa_records(Path("datasets/raw/chartqa/extracted"), excluded_chart_groups)
    knowledge = fvqa_records(
        Path("datasets/raw/fvqa/fvqa_train.parquet"),
        Path("datasets/formal/images/fvqa"),
        excluded_fvqa_ids,
        cache,
    )
    records = perception + knowledge
    if len(perception) != CHARTQA_LIMIT or len(knowledge) != FVQA_LIMIT:
        raise RuntimeError(f"prepared {len(perception)} perception and {len(knowledge)} knowledge records")
    write_jsonl(Path("datasets/manifests/formal_sft.jsonl"), records)
    print(f"wrote {len(records)} trajectories: {len(perception)} perception, {len(knowledge)} knowledge")


if __name__ == "__main__":
    main()
