#!/usr/bin/env python
"""Tiny action-aware SFT smoke: one real image, one masked response, few updates."""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="_pdf_qa/page-1.png")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default="artifacts/sft-smoke")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.config.use_cache = False
    model.visual.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    model.train()

    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Choose the next action for this document."},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = 'CROP {"bbox": [0.1, 0.1, 0.9, 0.9]}\nANSWER {"evidence_id": "crop_0"}'
    encoded = processor(text=[prompt], images=[image], return_tensors="pt")
    response_ids = processor.tokenizer(response, add_special_tokens=False, return_tensors="pt").input_ids
    prompt_len = encoded.input_ids.shape[-1]
    input_ids = torch.cat([encoded.input_ids, response_ids], dim=-1)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}
    for key in ("pixel_values", "image_grid_thw"):
        if key in encoded:
            batch[key] = encoded[key]
    batch = {key: value.to(device) for key, value in batch.items()}

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(json.dumps({"step": step + 1, "loss": losses[-1]}))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    print(json.dumps({
        "steps": args.steps,
        "losses": losses,
        "response_tokens": int(response_ids.shape[-1]),
        "mask_tokens": int((labels != -100).sum()),
        "loss_decreased": losses[-1] < losses[0],
        "checkpoint": str(output),
    }))


if __name__ == "__main__":
    main()
