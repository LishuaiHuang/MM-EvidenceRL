# SFT overfit smoke

记录时间：2026-09-15（Asia/Shanghai）  
状态：`OBSERVED`，仅验证训练链路，不代表模型效果

命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n reflectagent-grpo \
  python scripts/sft_overfit_smoke.py \
  --model /amax/home/lishuai/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --image _pdf_qa/page-1.png \
  --output artifacts/sft-smoke \
  --steps 3
```

配置为单张仓库图片、单条 action-aware response、fast processor、冻结视觉编码器和 LoRA 语言侧更新。结果：

```json
{
  "losses": [3.4265608788, 3.2971410751, 2.9962277412],
  "response_tokens": 37,
  "mask_tokens": 37,
  "loss_decreased": true,
  "trainable_params": 3686400
}
```

LoRA adapter 已保存并可通过 `PeftConfig.from_pretrained` 读取。生成的 `artifacts/sft-smoke/` 只作为本地可恢复 smoke checkpoint，不进入 Git。

这一步只证明 processor、视觉输入、response mask、反向传播、优化器更新和 checkpoint 保存链路可运行；下一步才是用数据节点提供的正式 manifest 做 200 条 SFT overfit 和 held-out schema 检查。
