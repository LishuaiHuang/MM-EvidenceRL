# MM-EvidenceRL Repository Instructions

These instructions apply to the entire repository.

## Engineering Style

- Write for the expected path first. Keep control flow linear and readable.
- Trust validated internal inputs. Do not repeat boundary checks between internal functions.
- Fail fast. Do not catch broad exceptions or silently fall back.
- Keep functions focused and implementations small.
- Prefer the standard library and established project dependencies.
- Do not add interfaces, factories, registries, or generic abstractions for a single implementation.
- Do not refactor unrelated files or add optional features without a current experiment requiring them.
- Comments should explain non-obvious decisions, not narrate ordinary code.

## Explicitly Out Of Scope

- Checksums, cryptographic hashes, or perceptual hashes for data verification.
- Retry loops, circuit breakers, automatic recovery systems, or multi-layer fallback frameworks.
- Premature streaming, chunking, or memory guards that have not been motivated by profiling.
- Mock-data generators or synthetic test harnesses inside production modules.
- Broad defensive validation for values already guaranteed by the manifest or environment.

Use small unit tests for deterministic logic such as action parsing, coordinate transforms, reward ordering, and citation lookup. Keep test fixtures outside production code.

## Project Boundaries

- Main policy: Qwen2.5-VL 3B class model.
- Visible actions: `CROP`, `SEARCH`, and terminal `ANSWER`.
- First version crops only from the root image and cites an `evidence_id`.
- Rollout keeps the visual encoder online for dynamic crops; vision parameters are initially frozen.
- Perception tasks mask `SEARCH`; knowledge tasks allow at most one search.
- Do not add PDF page navigation, nested crops, or explicit `GROUND`, `READ`, `VERIFY`, or `RECALL` tools to the first version.
- Do not claim experimental results until raw per-sample outputs and evaluator summaries exist.

## Two-Server Responsibilities

### Compute node: 8 x 48 GB

- Own model compatibility, multimodal rollout, SFT/GRPO, profiling, and active checkpoints.
- Keep only the current stage's prepared shards and one resumable checkpoint locally.
- Do not download or retain complete raw datasets or the full search corpus.

### Data node: large-disk server

- Own raw datasets, preprocessing, manifests, OCR/layout metadata, local search indices, prepared shards, and checkpoint archives.
- Do not start full dataset downloads before sources, licenses, split keys, and required subsets are recorded.
- Generated data and model artifacts stay outside Git. Commit only code, configs, manifests when suitably small, and reports.

## Git Coordination

- Compute-node work uses branches prefixed with `compute/`.
- Data-node work uses branches prefixed with `data/`.
- Do not have both nodes edit the same file concurrently.
- Before starting, run `git status --short` and `git pull --ff-only` on a clean branch.
- Make small commits that contain one coherent change. Do not commit secrets, tokens, passwords, raw datasets, caches, or checkpoints.
- Integration into `main` happens only after reviewing the changed files and the commands actually run.

## Current Gate

The immediate milestone is E0, not training:

1. Record both server environments without installing or upgrading the ML stack.
2. Select compatible model, processor, Transformers, vLLM, and veRL/Ray versions.
3. Implement and replay one `image -> CROP -> ANSWER` trajectory.
4. Implement and replay one `image -> SEARCH -> ANSWER` trajectory.
5. Profile actual visual tokens, GPU peak memory, host RSS, and disk use.

Do not generate 6,000 SFT trajectories or launch GRPO until E0 passes.
