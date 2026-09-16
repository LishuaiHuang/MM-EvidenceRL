#!/usr/bin/env python
"""Generate small Teacher and Recovery SFT trajectories in the local environment."""

import argparse
import json
import sqlite3
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


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
    return (
        "You are a multimodal evidence agent. Return exactly one JSON action. "
        "Allowed actions: CROP with bbox [x0,y0,x1,y1], SEARCH with query, or "
        "ANSWER with answer and evidence_id. Do not add markdown.\n"
        f"Question: {record['question']}\n"
        f"Environment history: {json.dumps(history, ensure_ascii=False)}\n"
        f"Environment feedback: {json.dumps(injected, ensure_ascii=False)}"
    )


def parse_action(raw):
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1])


def run_record(record, model, processor, device, args, trajectory_type):
    image = Image.open(args.image_root / record["image_path"]).convert("RGB")
    current_image = image
    history = []
    injected = {}
    if trajectory_type == "recovery":
        width, height = image.size
        wrong = image.crop((0, 0, max(1, width // 5), max(1, height // 5)))
        current_image = wrong
        history.append({
            "action": "CROP",
            "bbox": [0.0, 0.0, 0.2, 0.2],
            "status": "failed",
            "evidence_id": f"{record['sample_id']}:recovery_bad_crop",
        })
        injected = {"tool": "CROP", "status": "failed", "message": "The crop did not contain the answer. Choose another action."}
    for _ in range(args.max_steps):
        raw = model_step(model, processor, current_image, action_prompt(record, history, injected), device, args.max_new_tokens)
        action = parse_action(raw)
        name = action["action"]
        event = {"action": name, "raw": raw}
        if name == "CROP":
            event["bbox"] = action["bbox"]
            width, height = image.size
            x0, y0, x1, y1 = action["bbox"]
            current_image = image.crop((int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)))
            injected = {"tool": "CROP", "status": "ok", "evidence_id": f"{record['sample_id']}:teacher_crop:{len(history)}"}
        elif name == "SEARCH":
            result = search(args.search_index, record["sample_id"], action["query"])
            if result is None:
                injected = {"tool": "SEARCH", "status": "no_result"}
            else:
                injected = {"tool": "SEARCH", "status": "ok", **result}
        elif name == "ANSWER":
            event["answer"] = action["answer"]
            event["evidence_id"] = action.get("evidence_id")
            history.append(event)
            return {**record, "trajectory_type": trajectory_type, "trajectory": history,
                    "teacher_raw_actions": [item["raw"] for item in history],
                    "recovery_injected": trajectory_type == "recovery",
                    "recovered": trajectory_type != "recovery" or any(
                        item.get("status") == "ok" for item in history if item["action"] == "CROP"
                    )}
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
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    records = [record for record in records if record["partition"] == "sft_train"][:args.limit]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for index, record in enumerate(records):
            trajectory_type = "teacher" if index % 2 == 0 else "recovery"
            generated = run_record(record, model, processor, device, args, trajectory_type)
            handle.write(json.dumps(generated, ensure_ascii=False) + "\n")
            print(json.dumps({"sample_id": record["sample_id"], "trajectory_type": trajectory_type}))


if __name__ == "__main__":
    main()
