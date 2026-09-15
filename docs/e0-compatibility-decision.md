# E0 兼容性决策记录

记录时间：2026-09-15（Asia/Shanghai）

## 当前边界

本记录基于两台服务器的只读环境盘点和导入探针。没有安装或升级依赖，没有下载模型或数据，也没有把任何组合标记为已通过 E0。

## 观测结果

| 环境 | Python | PyTorch/CUDA | Transformers | vLLM | Ray | veRL | Qwen2.5-VL HF 导入 | vLLM 架构注册 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `reflectagent-executor` | 3.10.20 | 2.6.0+cu124 / 12.4 | 4.57.3 | 0.8.5.post1 | 2.51.2 | - | 通过 | 通过 |
| `reflectagent-grpo` | 3.10.20 | 2.9.0+cu128 / 12.8 | 4.57.3 | 0.11.1 | 2.51.2 | - | 通过 | 通过 |
| `searchr1` | 3.9.25 | 2.4.0+cu121 / 12.1 | 4.47.1 | 0.6.3 | 2.51.2 | 0.1 | 失败：Transformers 无 `Qwen2_5_VLForConditionalGeneration` | 未测试 |

两台节点的 GPU 均可被 PyTorch 访问，计算节点报告 8 张 RTX 4090、单卡约 48 GiB 显存。计算节点驱动报告 CUDA 13.0，实际候选 PyTorch runtime 为 CUDA 12.4 或 12.8；这属于驱动向后兼容条件，仍需模型加载探针确认。

## 暂定选择

- **主线多模态 runtime 候选：** `reflectagent-grpo`（Python 3.10、PyTorch 2.9.0+cu128、Transformers 4.57.3、vLLM 0.11.1、Ray 2.51.2、`qwen-vl-utils` 0.0.14）。它同时通过了 HF 模型类导入和 vLLM Qwen2.5-VL 架构注册。
- **回退 runtime 候选：** `reflectagent-executor`（PyTorch 2.6.0+cu124、vLLM 0.8.5.post1）。它同样通过了模型类导入和架构注册，但没有 `qwen-vl-utils`，需要在后续方案批准后单独评估是否补齐。
- **不作为主线：** `searchr1` 保留 veRL 0.1，但 Transformers 4.47.1 不支持目标模型，不能直接承担 Qwen2.5-VL rollout。

这只是候选基线，不是兼容性结论。当前还没有验证权重加载、processor 图像处理、多轮动态 crop、动作 logprob 或 learner 端接口。

## E0 下一步

在确定模型缓存绝对路径和允许的模型 revision 后，在主线候选环境执行单卡探针：

1. 加载 Qwen2.5-VL-3B 的 processor 和模型，记录实际磁盘增长、Host RSS、GPU 峰值显存和视觉 token 数。
2. replay `image -> CROP -> ANSWER` 与 `image -> SEARCH -> ANSWER` 两条最小 trajectory。
3. 检查 vLLM/Ray 的多模态输入、action token logprob、response mask 和确定性 replay。
4. 若主线候选在任一项失败，再切换回退环境；不得在未记录失败原因前混用两个环境。

模型加载探针需要用户确认模型缓存路径和下载权限后才能开始。
