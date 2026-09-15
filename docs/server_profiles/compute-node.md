# Compute Node Profile

盘点时间：2026-09-15 20:34 (Asia/Shanghai)

## Scope

本报告只记录计算节点的只读环境盘点结果。盘点期间未下载模型或数据，未安装或升级依赖，也未运行训练任务。

## Repository

```text
path: /amax/home/lishuai/MM-EvidenceRL
branch: compute/bootstrap
worktree: clean at start of profiling
```

## Operating System and Host

```text
OS: Ubuntu 20.04.1 LTS (focal)
kernel: 5.4.0-42-generic x86_64
CPU: 2 x Intel Xeon Gold 6530
logical CPUs: 128 (64 physical cores, SMT enabled)
NUMA nodes: 2
RAM: 125 GiB total, 28 GiB used, 93 GiB available at sampling time
swap: none
/dev/shm: 63 GiB total, 1.8 GiB used at sampling time
```

The project plan describes this as an 8 x 48 GB / 100 GB-class training node. The live machine reports 125 GiB RAM and no swap; the available root filesystem is the more immediate storage constraint.

## Storage

The root filesystem is the only local block device visible to this account:

```text
/dev/sda2  879G total  689G used  146G available  83%  /
/dev/sda1  976M total  7.8M used  967M available  /boot/efi
```

The node must not retain complete raw datasets, full search corpora, or multiple active checkpoints locally. At the current free space, model cache, current-stage shards, logs, and one resumable checkpoint need an explicit storage budget before E0 or training.

## GPU and Driver

`nvidia-smi` sees all eight devices and they were idle during profiling:

```text
driver: 580.126.09
reported CUDA compatibility: 13.0
GPU count: 8
GPU model: NVIDIA GeForce RTX 4090 (all devices)
memory: 49140 MiB per GPU
compute capability: 8.9
power limit: 450 W per GPU
MIG: not supported / not enabled
ECC: disabled (consumer GPU)
```

The GPUs are split across two NUMA nodes. `nvidia-smi topo -m` reports `NODE` links within each four-GPU group and `SYS` links across groups; no NVLink links are reported. The topology should inform process placement and any distributed training benchmark.

At profile time each GPU had approximately 18 MiB allocated to Xorg and 0% utilization. No training process was running.

## CUDA Tooling

Two CUDA toolkits are visible:

```text
nvcc: /amax/home/lishuai/cuda-12.1/bin/nvcc
nvcc release: 12.1, V12.1.105
system CUDA toolkit: /usr/local/cuda-13.0 (13.0.20251003)
```

The driver is newer than both toolkits. The project environment must select one toolkit/runtime path deliberately; `nvidia-smi`'s CUDA 13.0 line is driver compatibility information, not proof that a Python package was built for CUDA 13.

## Python and ML Environments

The active `base` environment is Python 3.13.13 and has none of the required ML packages installed. Other pre-existing environments contain incompatible partial stacks:

| Conda environment | Python | torch | Transformers | vLLM | Ray | DeepSpeed | datasets | veRL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base` | 3.13.13 | - | - | - | - | - | - | - |
| `dai` | 3.9.25 | 2.5.0+cu124 | 4.46.2 | - | - | - | - | - |
| `polarfree` | 3.10.20 | 2.6.0+cu124 | - | - | - | - | - | - |
| `reflectagent-data` | 3.11.15 | - | - | - | - | - | - | - |
| `reflectagent-executor` | 3.10.20 | 2.6.0+cu124 | 4.57.3 | 0.8.5.post1 | 2.51.2 | 0.16.9 | 3.6.0 | - |
| `reflectagent-grpo` | 3.10.20 | 2.9.0 | 4.57.3 | 0.11.1 | 2.51.2 | 0.16.9 | 3.6.0 | - |
| `retriever` | 3.10.20 | 2.4.0 | 4.47.1 | - | - | - | 5.0.0 | - |
| `searchr1` | 3.9.25 | 2.4.0+cu121 | 4.47.1 | 0.6.3 | 2.51.2 | - | 4.5.0 | 0.1 |

Additional package observations:

- `reflectagent-executor` has `accelerate 1.10.1` and `peft 0.17.1`, but no `verl`, `flash-attn`, or `qwen-vl-utils`.
- `reflectagent-grpo` has `accelerate 1.10.1`, `peft 0.17.1`, and `qwen-vl-utils 0.0.14`, but no `verl` or `flash-attn`.
- `searchr1` has `verl 0.1`, `flash-attn 2.8.3.post1`, and `accelerate 1.10.1`.

The existing environments are evidence for possible starting points only. They are not a validated MM-EvidenceRL compatibility matrix. In particular, no environment currently combines the target Qwen2.5-VL path, a verified processor, a verified multi-turn multimodal rollout, and a matching learner stack.

## E0 Compatibility Risks

1. **No single validated baseline environment.** The required packages are split across `reflectagent-*` and `searchr1`, with different Python, PyTorch, Transformers, vLLM, and CUDA builds.
2. **CUDA runtime mismatch risk.** The host exposes CUDA 13.0 libraries and CUDA 12.1 tooling, while installed PyTorch builds report CUDA 12.1 or 12.4, and one environment reports torch 2.9.0 without a CUDA suffix. A small import/device probe in the selected environment is required before model work.
3. **Rollout framework uncertainty.** `verl` exists only in `searchr1` at version 0.1; vLLM versions range from 0.6.3 to 0.11.1. Dynamic multi-image input, action logprob, response masking, and replay must be tested against the exact selected combination.
4. **Topology and memory bandwidth.** Four GPUs are local to each NUMA node and cross-group communication traverses `SYS`; process placement and NCCL behavior need a small distributed probe before multi-GPU training.
5. **Host memory and storage.** There is no swap, only about 93 GiB available RAM at sampling time, and 146G free on `/`. The current E0 gate should record process RSS and storage growth rather than assume the nominal hardware specification.

## Recommended E0 Order

1. Select one existing environment for a non-mutating probe, or create a project environment only after the compatibility decision is approved.
2. Verify `torch.cuda.device_count() == 8`, per-device names/capability, CUDA runtime, and a one-step tensor operation.
3. Run a single-GPU Qwen2.5-VL-3B processor/model load probe without downloading anything until the model cache path and disk budget are approved.
4. Test the two required trajectory shapes: `image -> CROP -> ANSWER` and `image -> SEARCH -> ANSWER`.
5. Test the exact rollout/learner path for multimodal inputs, action-token logprob, tool-response masking, and deterministic replay.
6. Measure visual tokens, GPU peak memory, host RSS, latency, and disk use before selecting a backend or starting SFT data generation.

## Commands Run

The profile used read-only commands including:

```text
pwd
git status --short --branch
df -h
free -h
nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-gpu=...
nvidia-smi -q -d ECC,POWER,TEMPERATURE,UTILIZATION
python3 --version
conda info --envs
conda list
conda run ... python -c ...
nvcc --version
uname -a
cat /etc/os-release
lscpu
df -h /dev/shm
lsblk
```

No credentials, tokens, full environment-variable dumps, model weights, datasets, caches, or checkpoints were recorded.
