#!/usr/bin/env python
"""Load Qwen2.5-VL once and run one image generation for E0 profiling."""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def rss_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS is unavailable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, use_fast=True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    load_seconds = time.perf_counter() - started

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the main content of this image briefly."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device)
    visual_shape = {
        name: list(value.shape)
        for name, value in inputs.items()
        if torch.is_tensor(value) and name in {"pixel_values", "image_grid_thw"}
    }
    visual_tokens = int(
        inputs.image_grid_thw.prod(dim=1).sum().item()
        / processor.image_processor.merge_size**2
    )
    input_tokens = int(inputs.input_ids.shape[-1])
    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    new_tokens = generated[:, input_tokens:]
    answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    torch.cuda.synchronize(device)

    result = {
        "model": args.model,
        "revision": args.revision,
        "image": os.path.abspath(args.image),
        "image_size": list(image.size),
        "load_seconds": round(load_seconds, 3),
        "input_tokens": input_tokens,
        "visual_tokens": visual_tokens,
        "generated_tokens": int(new_tokens.shape[-1]),
        "visual_inputs": visual_shape,
        "gpu_peak_allocated_mb": round(
            torch.cuda.max_memory_allocated(device) / 1024**2, 1
        ),
        "gpu_peak_reserved_mb": round(
            torch.cuda.max_memory_reserved(device) / 1024**2, 1
        ),
        "host_rss_mb": round(rss_mb(), 1),
        "answer": answer,
    }
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
