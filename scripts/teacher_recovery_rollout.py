#!/usr/bin/env python
"""Generate small Teacher and Recovery SFT trajectories in the local environment."""

import argparse
import json
import sqlite3
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def model_step(model, processor, image, prompt, device, max_new_tokens):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}
    ]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0].strip()


def search(index_path, sample_id, query):
    connection = sqlite3.connect(index_path)
    row = connection.execute(
        "SELECT evidence_id, title FROM evidence WHERE sample_id = ? ORDER BY rank LIMIT 1",
        (sample_id,),
    ).fetchone()
    connection.close()
    return {"evidence_id": row[0], "title": row[1], "query": query} if row else None


def action_prompt(record, history, injected):
    evidence_ready = any(item.get("status") == "ok" for item in history)
    if evidence_ready:
        route = "Evidence is available. The next action must be ANSWER using that evidence_id."
    elif record["task_mode"] == "perception":
        route = "The next useful action must be CROP on the root image. Do not ANSWER before CROP."
    elif record["requires_search"]:
        route = "The next useful action must be SEARCH. Do not ANSWER before SEARCH."
    else:
        route = "This task is search-free; ANSWER directly using the root image."
    return (
        "You are a multimodal evidence agent. Return exactly one JSON action. "
        "Allowed actions: CROP with bbox [x0,y0,x1,y1], SEARCH with query, or "
        "ANSWER with answer and evidence_id. Bbox coordinates are normalized from 0 to 1. "
        "Do not add markdown or explanations.\n"
        f"{route}\n"
        f"Question: {record['question']}\n"
        f"Environment history: {json.dumps(history, ensure_ascii=False)}\n"
        f"Environment feedback: {json.dumps(injected, ensure_ascii=False)}"
    )


def parse_action(raw):
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"teacher output is not JSON: {raw!r}")
    action, _ = json.JSONDecoder().raw_decode(raw[start:])
    return action


def normalized(value):
    return " ".join(str(value).strip().lower().split())


def crop_bbox(value, width, height):
    bbox = [float(item) for item in value]
    if max(bbox) > 1:
        bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
    bbox = [max(0.0, min(1.0, item)) for item in bbox]
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid crop bbox: {value!r}")
    return bbox


def run_record(record, model, processor, device, args, trajectory_type):
    image = Image.open(args.image_root / record["image_path"]).convert("RGB")
    current_image = image
    history = []
    raw_actions = []
    current_evidence = record["answer_evidence_id"] if not record["requires_search"] and record["task_mode"] == "knowledge" else None
    injected = {}
    if trajectory_type == "recovery":
        if record["task_mode"] == "knowledge" and not record["requires_search"]:
            raise ValueError(f"cannot inject recovery into a direct-answer record: {record['sample_id']}")
        width, height = image.size
        wrong = image.crop((0, 0, max(1, width // 5), max(1, height // 5)))
        current_image = image
        history.append({
            "action": "CROP",
            "bbox": [0.0, 0.0, 0.2, 0.2],
            "status": "failed",
            "evidence_id": f"{record['sample_id']}:recovery_bad_crop",
        })
        if record["task_mode"] == "knowledge":
            history[-1] = {
                "action": "SEARCH",
                "query": "irrelevant recovery query",
                "status": "failed",
                "evidence_id": f"{record['sample_id']}:recovery_bad_search",
            }
            injected = {"tool": "SEARCH", "status": "failed", "message": "No supporting result. Search again with a useful query."}
        else:
            x0, y0, x1, y1 = record["gold_bboxes_root"][0]
            injected = {
                "tool": "CROP",
                "status": "failed",
                "message": "The crop did not contain the answer. Choose another action.",
                "recovery_hint_bbox": [x0 / width, y0 / height, x1 / width, y1 / height],
            }
    for _ in range(args.max_steps):
        raw = model_step(model, processor, current_image, action_prompt(record, history, injected), device, args.max_new_tokens)
        action = parse_action(raw)
        raw_actions.append(raw)
        name = action["action"]
        event = {"action": name}
        if name == "CROP":
            if record["task_mode"] != "perception" or current_evidence is not None:
                if trajectory_type != "recovery":
                    raise ValueError(f"invalid CROP route for {record['sample_id']}")
                event["status"] = "failed"
                injected = {"tool": "CROP", "status": "failed", "message": "CROP is not allowed after evidence. Answer now."}
                history.append(event)
                continue
            width, height = image.size
            event["bbox"] = crop_bbox(action["bbox"], width, height)
            x0, y0, x1, y1 = event["bbox"]
            current_image = image.crop((int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)))
            current_evidence = f"{record['sample_id']}:teacher_crop:{len(history)}"
            event["evidence_id"] = current_evidence
            event["status"] = "ok"
            injected = {"tool": "CROP", "status": "ok", "evidence_id": current_evidence}
        elif name == "SEARCH":
            if record["task_mode"] != "knowledge" or current_evidence is not None:
                if trajectory_type != "recovery":
                    raise ValueError(f"invalid SEARCH route for {record['sample_id']}")
                event["status"] = "failed"
                injected = {"tool": "SEARCH", "status": "failed", "message": "SEARCH is not allowed. Use CROP or ANSWER according to the task."}
                history.append(event)
                continue
            result = search(args.search_index, record["sample_id"], action["query"])
            if result is None:
                raise ValueError(f"search returned no evidence for {record['sample_id']}")
            else:
                current_evidence = result["evidence_id"]
                event["query"] = action["query"]
                event["evidence_id"] = current_evidence
                event["status"] = "ok"
                injected = {"tool": "SEARCH", "status": "ok", **result}
        elif name == "ANSWER":
            if current_evidence is None:
                if trajectory_type != "recovery":
                    raise ValueError(f"ANSWER before evidence for {record['sample_id']}")
                event["status"] = "failed"
                injected = {"tool": "ANSWER", "status": "failed", "message": "No valid evidence yet. Recover with the required tool."}
                history.append(event)
                continue
            event["answer"] = action["answer"]
            event["evidence_id"] = action.get("evidence_id")
            if event["evidence_id"] != current_evidence:
                if trajectory_type != "recovery":
                    raise ValueError(f"ANSWER cited unknown evidence for {record['sample_id']}")
                event["status"] = "failed"
                injected = {"tool": "ANSWER", "status": "failed", "message": f"Cite the current evidence_id {current_evidence}."}
                history.append(event)
                continue
            if normalized(event["answer"]) != normalized(record["answer"]):
                if trajectory_type != "recovery":
                    raise ValueError(f"teacher answer disagrees for {record['sample_id']}: {event['answer']!r}")
                event["status"] = "failed"
                injected = {"tool": "ANSWER", "status": "failed", "message": "The answer is not supported by the current evidence. Re-read the evidence and answer again."}
                history.append(event)
                continue
            history.append(event)
            recovered = trajectory_type != "recovery" or any(
                item.get("status") == "ok" for item in history if item["action"] in {"CROP", "SEARCH"}
            )
            if not recovered:
                raise RuntimeError(f"teacher did not recover sample {record['sample_id']}")
            return {**record, "trajectory_type": trajectory_type, "trajectory": history,
                    "teacher_raw_actions": raw_actions,
                    "teacher_model": args.model,
                    "recovery_injected": trajectory_type == "recovery",
                    "recovery_guided": trajectory_type == "recovery" and record["task_mode"] == "perception",
                    "recovered": recovered}
        else:
            injected = {"error": "invalid_action", "message": "Use CROP, SEARCH, or ANSWER."}
        event["feedback"] = injected
        history.append(event)
    raise RuntimeError(f"teacher did not answer sample {record['sample_id']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/manifests/formal_sft_split.jsonl")
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--search-index", default="datasets/indices/formal_search/evidence.sqlite3")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output", default="artifacts/teacher-recovery-sft.jsonl")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--scan-limit", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    train = [record for record in records if record["partition"] == "sft_train"]
    teachers = [record for record in train if record["task_mode"] == "perception"]
    recoveries = [record for record in train if record["task_mode"] == "perception"]
    records = [item for pair in zip(teachers, recoveries) for item in pair][:args.scan_limit]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        written = 0
        for index, record in enumerate(records):
            if written == args.limit:
                break
            trajectory_type = "teacher" if index % 2 == 0 else "recovery"
            try:
                generated = run_record(record, model, processor, device, args, trajectory_type)
            except (ValueError, RuntimeError, json.JSONDecodeError) as error:
                print(json.dumps({"sample_id": record["sample_id"], "trajectory_type": trajectory_type, "rejected": str(error)}))
                continue
            handle.write(json.dumps(generated, ensure_ascii=False) + "\n")
            print(json.dumps({"sample_id": record["sample_id"], "trajectory_type": trajectory_type}))
            written += 1
        if written < args.limit:
            raise RuntimeError(f"only generated {written} valid trajectories after scanning {len(records)} records")


if __name__ == "__main__":
    main()
