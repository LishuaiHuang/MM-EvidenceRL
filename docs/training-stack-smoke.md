# 训练栈最小 smoke 记录

记录时间：2026-09-15（Asia/Shanghai）  
状态：`OBSERVED`

## 冻结配置

- 环境：`reflectagent-grpo`
- Python：3.10.20
- PyTorch：2.9.0+cu128
- Transformers：4.57.3
- vLLM：0.11.1
- Ray：2.51.2（仅保留为后续 worker 承载，不在此阶段引入 veRL）
- 模型：`Qwen/Qwen2.5-VL-3B-Instruct@main`
- Processor：`Qwen2VLImageProcessorFast`

vLLM 的 Qwen2.5-VL processor 会忽略 `use_fast=False`，因此 HF 和 vLLM 统一使用 fast processor。视觉 token、prompt template 和 processor 参数必须由 rollout 与 learner 共用同一 snapshot。

## HF logprob / response mask smoke

脚本：`scripts/logprob_mask_probe.py`

它执行一次多模态 teacher-forced full re-prefill，返回 response 区间的逐 token logprob，并构造只覆盖 response 的布尔 mask。当前最小结果：

```json
{
  "task_mode": "perception",
  "prompt_tokens": 2802,
  "response_tokens": 43,
  "response_mask_tokens": 43,
  "mean_response_logprob": -3.078125,
  "search_allowed": false,
  "search_count": 0,
  "search_mask_valid": true,
  "processor": "Qwen2VLImageProcessorFast"
}
```

同脚本的 knowledge 模式允许一次 SEARCH，并得到 `search_count=1`、`search_mask_valid=true`。这验证了 SFT/GRPO learner 所需的最小 response-mask 和任务动作约束接口。

## vLLM token logprob smoke

vLLM 单图 `LLM.generate(..., SamplingParams(logprobs=5))` 已返回生成序列的 token logprob 条目（最小请求得到 12 条），并成功生成 `CROP/SEARCH/ANSWER` 相关文本。由于当前仓库没有 veRL/learner 实现，不额外搭建 Ray/veRL 编排层；GRPO 第一版采用 vLLM rollout + HF full re-prefill learner，二者共享 fast processor 和同一 chat template。

## 可省略项

- 不做多模态增量 KV cache；每轮 full re-prefill 已足够支持第一版动态 crop。
- 不做 veRL 原生适配，直到 learner 接口实际需要；`searchr1` 的 veRL 0.1 与目标 Transformers 不兼容。
- 不做 200 条长稳态或多节点压测作为进入 SFT 的前置条件；先用小规模 SFT smoke 暴露实际 OOM/worker 问题。

## 仍需在训练中观察

- rollout token logprob 与 learner 重算值的逐 token 差异；
- SFT action schema 合法率和 mask 覆盖范围；
- 多 GPU learner 的 NCCL/显存稳定性。
