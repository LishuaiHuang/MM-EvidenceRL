#!/usr/bin/env python
"""Replay the two minimal E0 trajectories with one loaded Qwen2.5-VL model."""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def generate(model, processor, image, text, device, max_new_tokens):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": text}
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    input_tokens = int(inputs.input_ids.shape[-1])
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = processor.batch_decode(output[:, input_tokens:], skip_special_tokens=True)[0].strip()
    visual_tokens = int(
        inputs.image_grid_thw.prod(dim=1).sum().item()
        / processor.image_processor.merge_size**2
    )
    return answer, input_tokens, visual_tokens


def search_line(path, query):
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if query in line:
            return f"{path}:{line_number}", line.strip()
    raise ValueError(f"query not found: {query}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--search-document", default="MM-EvidenceRL_实验计划.md")
    parser.add_argument("--search-query", default="模型可见动作只有")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, device_map="cuda:0"
    )
    root = Image.open(args.image).convert("RGB")
    width, height = root.size

    crop = root.crop((int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9)))
    started = time.perf_counter()
    crop_answer, crop_input, crop_visual = generate(
        model, processor, crop,
        "This is the CROP step. Summarize the visible document content briefly.",
        device, args.max_new_tokens,
    )
    crop_seconds = time.perf_counter() - started

    started = time.perf_counter()
    evidence_id, snippet = search_line(args.search_document, args.search_query)
    search_action = f"SEARCH(query={args.search_query!r})"
    evidence = f"evidence_id={evidence_id}; snippet={snippet}"
    search_answer, search_input, search_visual = generate(
        model, processor, root,
        f"This is the ANSWER step after {search_action}. Use this retrieved evidence: {evidence} Answer which actions are visible to the model.",
        device, args.max_new_tokens,
    )
    search_seconds = time.perf_counter() - started

    replay_answer, _, _ = generate(
        model, processor, root,
        f"This is the ANSWER step after {search_action}. Use this retrieved evidence: {evidence} Answer which actions are visible to the model.",
        device, args.max_new_tokens,
    )
    torch.cuda.synchronize(device)

    print(json.dumps({
        "model": args.model,
        "revision": args.revision,
        "image": args.image,
        "crop_trajectory": {
            "steps": ["CROP", "ANSWER"],
            "crop_bbox_root": [0.1, 0.1, 0.9, 0.9],
            "input_tokens": crop_input,
            "visual_tokens": crop_visual,
            "latency_seconds": round(crop_seconds, 3),
            "answer": crop_answer,
        },
        "search_trajectory": {
            "steps": ["SEARCH", "ANSWER"],
            "evidence_id": evidence_id,
            "snippet": snippet,
            "input_tokens": search_input,
            "visual_tokens": search_visual,
            "latency_seconds": round(search_seconds, 3),
            "answer": search_answer,
            "replay_match": search_answer == replay_answer,
        },
        "gpu_peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
