# 双节点兼容性与 E0 决策记录

状态：`PLAN / UNVERIFIED`。本记录只合并两台服务器的环境盘点，不代表模型、processor 或 rollout 已通过验证。

盘点依据：

- 计算节点报告：`origin/compute/bootstrap`，commit `32af390`
- 数据节点报告：`origin/data/bootstrap`，commit `edb69c9`

## 1. 资源边界

| 项目 | 数据节点 | 计算节点 | 对 E0 的含义 |
|---|---|---|---|
| GPU | 2× RTX 4090，24564 MiB/卡 | 8× RTX 4090，49140 MiB/卡 | 数据节点可做小规模视觉/OCR；模型 rollout/learner 在计算节点 |
| Host RAM | 251 GiB，总可用约 189 GiB | 125 GiB，总可用约 93 GiB | 预处理、索引和派生数据优先放数据节点 |
| 本地磁盘 | 约 2.1T 可用，86% 已用 | 约 146G 可用，83% 已用 | 计算节点只保留当前阶段 shard、缓存和一个 resumable checkpoint |
| Swap | 无 | 无 | 不把 Swap 作为容量方案，必须记录 RSS/磁盘增长 |
| Python | 3.8.5 | base 3.13.13；候选环境 3.9/3.10 | 计算节点需先选候选环境，数据节点不安装训练栈 |
| Torch | 2.4.1+cu124，CUDA 可见 2 卡 | 现有环境为 2.4.0+cu121、2.5.0/2.6.0+cu124、2.9.0 等 | 不把不同环境混用，先做单环境 probe |

## 2. 计算节点候选环境

以下是已存在环境的只读记录，不是已批准的依赖组合：

| 环境 | Python | Torch | Transformers | vLLM | Ray | DeepSpeed | datasets | veRL | 处理 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `searchr1` | 3.9.25 | 2.4.0+cu121 | 4.47.1 | 0.6.3 | 2.51.2 | - | 4.5.0 | 0.1 | 优先做 veRL 路径的只读模型/processor probe |
| `reflectagent-executor` | 3.10.20 | 2.6.0+cu124 | 4.57.3 | 0.8.5.post1 | 2.51.2 | 0.16.9 | 3.6.0 | - | 作为 Transformers/vLLM fallback 候选 |
| `reflectagent-grpo` | 3.10.20 | 2.9.0 | 4.57.3 | 0.11.1 | 2.51.2 | 0.16.9 | 3.6.0 | - | 仅在模型加载和 rollout 接口验证后考虑 |

当前没有任何候选环境同时证明了：目标 Qwen2.5-VL revision 可加载、动态 crop 多轮输入可用、action logprob 与 response mask 对齐、轨迹可 replay。因此下一步应由计算节点先做小规模兼容性 probe，不应先安装或升级整套依赖。

## 3. E0 后端决策顺序

1. `searchr1` 中验证 veRL + vLLM 的单卡/两轮多模态最小路径；
2. 若动态图片追加或 logprob/mask 不稳定，测试 full re-prefill；
3. 若 veRL 路径仍不稳定，使用已有 `reflectagent-executor` 或 `reflectagent-grpo` 做 Transformers/Ray fallback probe；
4. 只有在 `image → CROP → ANSWER` 和 `image → SEARCH → ANSWER` 都能记录完整 trajectory 后，才锁定 backend 和版本。

这不是安装授权，也不改变“先盘点、后决定”的当前闸门。

## 4. 数据节点配合项

数据节点在计算后端未锁定前，只准备不依赖具体训练框架的内容：

- 数据源/许可证登记；
- `group_id` 和 split 规则；
- E0 manifest 字段；
- root 坐标系 bbox 与 evidence 字段；
- 本地 Search 语料的来源审计；
- 小规模、可重放的样本子集。

不提前下载全量图片、完整 Wikipedia corpus、模型权重或训练 shard。
