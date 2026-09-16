# Teacher / Recovery rollout

`scripts/teacher_recovery_rollout.py` generates action-aware SFT trajectories by
running the frozen multimodal model against the local image and evidence
environment. It writes only trajectories that satisfy the action route, answer,
and evidence-id checks.

Use the data-node prepared roots after they are mounted on the compute node:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n reflectagent-grpo \
  python scripts/teacher_recovery_rollout.py \
  --manifest datasets/manifests/formal_sft_split.jsonl \
  --image-root /prepared/formal \
  --search-index /prepared/indices/formal_search/evidence.sqlite3 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --output artifacts/teacher-recovery-sft.jsonl \
  --limit 4
```

Perception records require `CROP → ANSWER`; searchable knowledge records
require `SEARCH → ANSWER`. Recovery starts with an injected failed tool event
and is retained only if a later tool call succeeds before the cited answer.
Malformed actions, unsupported routes, wrong answers, and unknown citations
fail the run rather than entering the SFT set. Generated JSONL remains local
under `artifacts/`.
