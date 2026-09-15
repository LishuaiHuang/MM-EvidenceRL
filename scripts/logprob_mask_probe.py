#!/usr/bin/env python
"""Minimal HF teacher-forced logprob and response-mask smoke test."""

import argparse
import json

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--task-mode", choices=["perception", "knowledge"], default="perception")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    image = Image.open(args.image).convert("RGB")
    user_text = "Choose the next tool action for this image."
    response = (
        'CROP {"bbox": [0.1, 0.1, 0.9, 0.9]}\nANSWER {"evidence_id": "crop_0", "text": "document"}'
        if args.task_mode == "perception"
        else 'SEARCH {"query": "document topic"}\nANSWER {"evidence_id": "search_0", "text": "document"}'
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": user_text}
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[prompt], images=[image], return_tensors="pt")
    prompt_len = int(encoded.input_ids.shape[-1])
    response_ids = processor.tokenizer(response, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([encoded.input_ids, response_ids], dim=-1).to(device)
    attention_mask = torch.ones_like(input_ids)
    model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if key in encoded:
            model_inputs[key] = encoded[key].to(device)
    with torch.inference_mode():
        logits = model(**model_inputs).logits
    target = input_ids[:, 1:]
    token_logprobs = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    response_logprobs = token_logprobs[:, prompt_len - 1:]
    response_mask = torch.ones_like(response_logprobs, dtype=torch.bool)
    search_allowed = args.task_mode == "knowledge"
    result = {
        "task_mode": args.task_mode,
        "prompt_tokens": prompt_len,
        "response_tokens": int(response_ids.shape[-1]),
        "response_mask_tokens": int(response_mask.sum().item()),
        "mean_response_logprob": round(float(response_logprobs.mean().item()), 6),
        "search_allowed": search_allowed,
        "search_count": response.count("SEARCH"),
        "search_mask_valid": search_allowed or "SEARCH" not in response,
        "processor": processor.image_processor.__class__.__name__,
    }
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
