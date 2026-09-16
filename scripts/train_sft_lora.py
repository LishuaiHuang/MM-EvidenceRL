#!/usr/bin/env python
"""Train the action policy with a memory-bounded LoRA SFT loop."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def response_text(record):
    return "\n".join(json.dumps(action, ensure_ascii=False, separators=(",", ":"))
                     for action in record["trajectory"]) + "\n"


def prompt_text(record):
    mode = record["task_mode"]
    route = "CROP then ANSWER" if mode == "perception" else (
        "SEARCH then ANSWER" if record["requires_search"] else "ANSWER"
    )
    return (
        "You are a multimodal evidence agent. Return one JSON action per line. "
        "Allowed actions are CROP with bbox, SEARCH with query, and ANSWER with answer and evidence_id. "
        f"For this {mode} task, use the route {route}.\n"
        f"Question: {record['question']}"
    )


def make_batch(record, processor, image_root, device):
    image = Image.open(image_root / record["image_path"]).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text(record)},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[prompt], images=[image], return_tensors="pt")
    response_ids = processor.tokenizer(
        response_text(record), add_special_tokens=False, return_tensors="pt"
    ).input_ids
    input_ids = torch.cat([encoded.input_ids, response_ids], dim=-1)
    labels = input_ids.clone()
    labels[:, :encoded.input_ids.shape[-1]] = -100
    batch = {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": torch.ones_like(input_ids, device=device),
    }
    for key in ("pixel_values", "image_grid_thw"):
        if key in encoded:
            batch[key] = encoded[key].to(device)
    return batch, int(response_ids.shape[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/sft-full/formal_sft.jsonl")
    parser.add_argument("--image-root", type=Path, default=Path("/amax/home/lishuai/prepared/formal"))
    parser.add_argument("--model", default="/amax/home/lishuai/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3")
    parser.add_argument("--output", default="artifacts/sft-lora")
    parser.add_argument("--logdir", default="runs/sft-lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.config.use_cache = False
    model.visual.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    writer = SummaryWriter(args.logdir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    if args.max_samples is not None:
        records = records[:args.max_samples]
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for record in records:
            batch, response_tokens = make_batch(record, processor, args.image_root, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            step += 1
            value = float(loss.detach().cpu())
            writer.add_scalar("train/loss", value, step)
            writer.add_scalar("train/response_tokens", response_tokens, step)
            if step == 1 or step % 25 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": value}), flush=True)
            if step % args.save_every == 0:
                checkpoint = output / f"checkpoint-{step}"
                model.save_pretrained(checkpoint)
                processor.save_pretrained(checkpoint)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    writer.close()
    print(json.dumps({"steps": step, "samples": len(records), "checkpoint": str(output)}))


if __name__ == "__main__":
    main()
