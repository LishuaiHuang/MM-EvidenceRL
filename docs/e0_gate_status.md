# E0 跨节点验收状态

更新时间：2026-09-15

## 结论

E0 最小闭环：`PASS`。

计算节点已在 `origin/compute/bootstrap` 完成并记录：

- Qwen2.5-VL-3B-Instruct 单卡加载和生成；
- root image 与动态 crop 的视觉编码；
- `image → CROP → ANSWER`；
- `image → SEARCH → ANSWER`，含可追溯 `evidence_id`；
- 相同输入的确定性 replay；
- vLLM 0.11.1 单卡多模态 serving；
- 视觉 token、峰值显存、Host RSS、延迟和模型缓存占用观测。

关键观测值：root 视觉 token 2772、crop 视觉 token 1750、HF 峰值显存约 7.73 GB、Host RSS 约 3.30 GB；vLLM 权重占用约 7.16 GiB，单图生成成功。

## 仍未验收的训练集成项

以下不是 E0 最小闭环失败，而是进入 SFT/GRPO 前的下一项兼容性工作：

- action logprob 与 response mask；
- Ray/learner 对接；
- 多 GPU 拓扑下的并行稳定性；
- 长序列、多轮 episode 的 RSS/OOM 观测；
- vLLM fast processor 与 HF slow processor 的视觉 token 配置统一。

## 下一步

1. 计算节点固定模型、processor、Transformers、vLLM 和 veRL/Ray 版本，完成上述训练集成探针。
2. 数据节点以 `datasets/manifests/e0.jsonl` 的字段协议为准，扩展正式来源的 SFT/GRPO 轨迹；不重复下载 MathVista，也不把它放入训练集。
3. 生成 6,000 条轨迹前，先完成来源 split/group 隔离和本地 evidence 索引方案；E0 样本本身不直接混入正式训练。
