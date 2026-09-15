# MM-EvidenceRL 实验计划

> 项目：基于主动视觉聚焦与反事实验证的多模态推理 Agent  
> 周期：10 周  
> 主模型：Qwen2.5-VL-3B 级别模型  
> 当前状态：实验设计已冻结，尚未产生训练结果  
> 主方案依据：[MM-EvidenceRL_平衡策略评估与项目借鉴方案.md](./MM-EvidenceRL_平衡策略评估与项目借鉴方案.md)

---

## 0. 实验契约

### 0.1 主线固定项

主线只实现并验证下面这条闭环：

```text
root image + question
    → CROP / SEARCH / ANSWER
    → environment evidence ledger
    → Action-aware SFT
    → multi-turn on-policy GRPO
    → answer / evidence / grounding / robustness / efficiency evaluation
```

冻结以下范围，除非第一周闸门测试证明不可实现：

- 主模型使用 3B，不把 7B 作为主交付；
- 模型可见动作只有 `CROP`、`SEARCH`、`ANSWER`；
- `perception` 与 `knowledge` 两类任务分流；
- 视觉编码器在 rollout 时在线运行，第一版训练时冻结其参数；
- root image 在单条 trajectory 内复用视觉特征，动态 crop 在线编码；
- 第一版只允许从 root image 发起 crop，不允许嵌套 crop；
- 模型在 `ANSWER` 中引用 `evidence_id`，环境负责映射到 root 坐标；
- SFT 轨迹总规模约 6,000 条；
- GRPO 问题单元 1,000 个，初始每题 `G=4` 条 rollout；
- sealed test 固定 800 题，perception/knowledge 各 400 题；
- 训练期约 15% 的可用 episode 构造反事实版本；
- 只把日志和 evaluator 能复查的数字写入 README、报告和简历。

### 0.2 不进入主线的内容

- 主动 PDF 翻页和任意页调度；
- 显式 `GROUND`、`READ`、`VERIFY`、`RECALL` 工具；
- 公网搜索 API；
- 嵌套 crop 和复杂局部坐标链；
- 未经 profiling 的跨 episode 视觉缓存；
- 为了展示框架名而坚持不稳定的多模态增量 KV cache；
- 没有可靠区域标注的样本参与 grounding reward；
- 用 sealed test 调超参数或选择 checkpoint。

### 0.3 实验结论的证据等级

所有结论分成三类，文档中不得混写：

| 标签 | 含义 | 允许的表述 |
|---|---|---|
| `PLAN` | 尚未运行 | “计划采用”“目标验证” |
| `OBSERVED` | 有原始日志，但未复现 | “单次实验观察到” |
| `VERIFIED` | 配置、日志、checkpoint 和 evaluator 可复查 | “实测”“提升/降低 X%” |

---

## 1. 成功标准与停止条件

### 1.1 基础设施成功标准

在进入正式 SFT 前必须满足：

1. 单图推理、动作解析、crop 执行、search 执行和 answer 终止全部可运行；
2. 同一条保存的 trajectory 能确定性 replay，环境状态和 evidence ledger 一致；
3. root/crop 的实际 visual token 数可记录且不会突破预算；
4. evidence citation 能稳定映射回 root 坐标并生成可视化框；
5. rollout 端和 learner 端使用相同的图片处理器、视觉预算和 action mask；
6. 连续运行 200 条两轮 episode 不发生 GPU OOM、Host OOM 或 worker desync；
7. 已确定主 rollout backend 及 fallback，不能带着兼容性未知项进入第 3 周。

### 1.2 学习有效性成功标准

最终不能只证明“系统能跑”，至少需要同时满足：

- SFT 后 held-out action JSON/schema 合法率不低于 98%；
- knowledge 验证集上的 Search 行为不坍缩为零，并且 Search Hit、引用有效率可单独统计；
- perception 样本中 Search 被环境 mask，不把无意义不搜索当作策略进步；
- GRPO 训练期间大多数有效 batch 存在非零 group reward 方差；
- 最终 GRPO 相比 SFT 在预先指定的主指标 `Joint Success` 上有正向趋势；
- 答案正确率、证据命中、grounding、遮挡鲁棒性和工具成本至少形成一组可解释的权衡，而不是只优化单一 reward；
- 至少完成一个无工具 Base、一个工具化 Base、SFT、GRPO 和关键消融；
- sealed test 只在模型选择和 reward 定型后运行。

其中建议将 `Joint Success` 固定为：

```text
perception_joint = answer_correct AND visual_evidence_hit AND citation_valid
knowledge_joint  = answer_correct AND search_evidence_hit AND citation_valid
```

Grounding Coverage/IoU 作为 joint 结果的分解指标报告，不用一个门槛覆盖所有 region type。

### 1.3 立即停止并回退的条件

- 第 3 天仍无法稳定获取多轮 action logprob 或 policy mask：切换 full re-prefill backend；
- 连续 200 条安全配置 episode 仍有 OOM：降低 crop token、并发和 prefetch，不进入 RL；
- 坐标可视化与人工标注不一致：停止所有 grounding reward 实验；
- reward 单元测试的排序关系不成立：停止训练，先修 evaluator；
- 数据 split 存在同图、同文档或同实体泄漏：重新生成 manifest；
- Search corpus 的 top-k 证据质量不够：先修索引，不用 GRPO 掩盖检索失败；
- 训练指标只有总 reward，没有分量和轨迹日志：该 run 不计入有效实验。

---

## 2. 目录、配置与复现规则

### 2.1 目标目录结构

下面是实施阶段的目标结构，不要求第一天一次性创建所有空目录：

```text
MM_EvidenceRL/
├─ configs/
│  ├─ env/
│  ├─ data/
│  ├─ sft/
│  ├─ grpo/
│  └─ eval/
├─ data/
│  ├─ raw/                 # 原始数据，不直接修改
│  ├─ interim/             # OCR、bbox、检索文档等中间产物
│  ├─ manifests/           # train/dev/grpo/sealed 清单
│  └─ indices/             # BM25/BGE-M3 本地索引
├─ mm_evidence_rl/
│  ├─ env/                 # action parser、crop/search、ledger
│  ├─ data/                # benchmark adapters
│  ├─ rollout/             # backend adapter、trajectory recorder
│  ├─ reward/              # reward components 与 gate
│  ├─ train/               # SFT/GRPO entrypoints
│  └─ eval/                # answer/grounding/robustness/system metrics
├─ scripts/
├─ tests/
├─ logs/
├─ checkpoints/
├─ artifacts/
│  ├─ replays/
│  ├─ overlays/
│  ├─ profiles/
│  └─ reports/
└─ docs/
```

`data/raw`、大模型权重、checkpoint 和完整日志不直接提交 Git；manifest、配置、评测代码和小型示例应提交。

### 2.2 每次 run 必须固化的信息

每次 SFT、GRPO 和正式评测都保存：

- `run_id`；
- Git commit 和 dirty-worktree 标记；
- resolved config；
- 模型名、revision、processor revision；
- 数据集版本、manifest 文件名/版本、搜索语料 snapshot id；
- CUDA、driver、PyTorch、Transformers、vLLM、veRL/Ray 版本；
- GPU 型号和数量、Host RAM；
- seed；
- 启动命令；
- stdout/stderr、标量日志和异常轨迹；
- checkpoint 与 evaluator 版本。

建议命名：

```text
{phase}-{model}-{data}-{reward}-{seed}-{yyyymmdd-hhmm}

例：
sft-qwen25vl3b-mix6k-action-v0-s42-20260922-2100
grpo-qwen25vl3b-mix1k-gated-v1-s42-20261013-0900
```

### 2.3 最小数据 schema

样本 manifest：

```json
{
  "sample_id": "chartqa_000123",
  "dataset": "chartqa",
  "split": "train",
  "group_id": "source_image_or_document_id",
  "task_mode": "perception",
  "image_path": "...",
  "question": "...",
  "answer": "...",
  "answer_type": "number",
  "region_type": "micro",
  "gold_bboxes_root": [[0.12, 0.28, 0.19, 0.34]],
  "requires_search": false,
  "search_evidence_ids": [],
  "counterfactual_available": true
}
```

trajectory JSONL：

```json
{
  "trajectory_id": "...",
  "sample_id": "...",
  "policy_revision": "...",
  "processor_revision": "...",
  "steps": [],
  "ledger": [],
  "visual_tokens": {"root": 0, "crops": [], "total": 0},
  "reward_components": {},
  "terminal_reason": "answer",
  "timing_ms": {},
  "gpu_peak_mb": 0,
  "host_rss_mb": 0
}
```

评测结果每题一行，不能只保存汇总均值：

```json
{
  "sample_id": "...",
  "run_id": "...",
  "answer_correct": 1,
  "evidence_hit": 1,
  "citation_valid": 1,
  "coverage": 0.95,
  "strict_iou": null,
  "tightness": 0.14,
  "tool_steps": 1,
  "crop_calls": 1,
  "search_calls": 0,
  "invalid_actions": 0,
  "counterfactual_outcome": null
}
```

---

## 3. Day 0 与第 1 周硬闸门

### Day 0：机器、依赖和存储基线

检查并保存：

- 8×48GB 节点和 2×24GB 节点的 GPU 拓扑、驱动、CUDA、NCCL；
- PyTorch 能否识别全部 GPU；
- 模型权重、数据、日志和 checkpoint 的可用磁盘空间；
- Host RAM、shared memory、DataLoader/pinned memory 默认值；
- 当前 Transformers、vLLM、veRL、Ray 对目标模型 revision 的支持情况；
- 单卡加载 3B 模型和 processor 的峰值显存；
- 开源代码、模型、数据集和搜索语料的 license/使用限制。

产物：

```text
artifacts/profiles/day0_system.json
artifacts/reports/dependency_matrix.md
artifacts/reports/license_inventory.md
```

### Day 1：单轮输入与协议

完成：

1. root image + question 单轮推理；
2. 三种 action JSON 的严格 parser；
3. 越界 bbox、空 query、未知 evidence id、重复 ANSWER 的错误处理；
4. 记录 processor 的 `image_grid_thw` 或等价字段和真实 visual token 数；
5. 保存完整 prompt、模型原始输出和解析后动作。

通过标准：20 个样例均能结束；非法动作返回结构化错误，不导致 worker 崩溃。

### Day 2：动态 Crop、坐标与 replay

完成：

1. 模型或 oracle 产生 root-normalized bbox；
2. 环境 crop、固定预算重采样并在线视觉编码；
3. ledger 写入 `crop_1` 和 root bbox；
4. 第二轮使用 root + crop 的完整上下文生成 ANSWER；
5. 引用 `crop_1` 后由 evaluator 恢复 canonical root bbox；
6. 输出 root image 上的预测框/Gold 框 overlay；
7. 保存 trajectory 后重新 replay。

通过标准：10 个手工 bbox 的像素坐标误差不超过 1 像素或归一化误差不超过 `1e-3`；replay 的 ledger、终止原因和 evaluator 输入完全一致。

### Day 3：veRL/vLLM 多轮多模态兼容性闸门

最小轨迹：

```text
root image → CROP → crop image observation → ANSWER
```

必须同时验证：

- 第二轮多图/动态图片输入；
- action token 的 generation logprob；
- tool response token 不进入 policy loss；
- response mask 与 token 序列对齐；
- learner 端重算 logprob 与 rollout 端差异在可解释范围内；
- 轨迹能进入一个最小 GRPO loss 前向，不要求当天训练收敛。

Backend 决策优先级：

1. veRL + vLLM 原生稳定路径；
2. vLLM 每轮 full re-prefill，不做多模态增量 KV append；
3. Ray + Transformers full re-prefill。

第 3 天结束必须记录“采用哪条路径、为什么、已知吞吐代价”。若第一条不稳定，立即用第二或第三条，不继续在底层 CUDA 上无限排查。

### Day 4：本地 Search 与任务分流

完成：

- 建立小型 BM25 索引；
- 如资源允许加入 BGE-M3 召回，再用 RRF 合并；
- knowledge 样本开放最多一次 Search；
- perception 样本在 action logits 或合法动作层 mask Search；
- 返回 top-k snippet 和稳定 evidence id；
- query 去重、空 query、超预算均有结构化结果。

通过标准：人工选取 20 个 knowledge 问题，至少 16 个问题的 top-5 中包含可支持答案的证据；未达到时先修语料和索引。

### Day 5：Reward 和 visual token profiling

对人工构造轨迹验证 reward 的相对排序：

```text
正确 + 精确有效证据
    > 正确 + 粗粒度/冗余证据
    > 正确 + 无证据
    > 错误 + 看似精确的框
    > 非法格式
```

同时测量：

- root budget 256/512；
- crop budget 128/256；
- 0/1/2 次 crop；
- `G=1/2/4`；
- 峰值 GPU 显存、Host RSS、首 token 延迟、episode latency；
- processor 实际视觉 token，而不是只记录目标分辨率。

通过标准：确定第一个安全配置，连续 200 条 episode 无 OOM；超限 crop 返回 `CROP_REJECTED(reason=visual_budget)`。

### Day 6：40 条端到端 smoke replay

- 20 条 perception；
- 20 条 knowledge；
- 同时包含 direct answer、crop、search 和错误恢复；
- 不训练，仅使用 oracle/人工轨迹和基础模型回放；
- 输出逐题 trajectory、ledger、reward、overlay 和系统 profile。

### Day 7：冻结环境 v0

评审并冻结：

- action schema；
- coordinate convention；
- visual token budget；
- rollout backend；
- trajectory schema；
- task mask；
- reward evaluator v0；
- 40 条 smoke 的失败清单。

退出标准：能展示并重放一条 `image → crop → answer+citation` 和一条 `image → search → answer+citation`，且日志足够定位每一步。

---

## 4. 数据实验计划

### 4.1 数据组成与用途

| 数据源 | 模式 | 建议占比 | 主要能力 | 是否默认 Search |
|---|---|---:|---|---|
| ChartQA | perception | 30% | 图表局部读取、数值推理 | 否 |
| DocVQA/InfoVQA | perception | 25% | 文档布局、小字、表格 | 否 |
| MathVista | perception | 15% | 视觉计算与符号推理 | 否 |
| InfoSeek/FVQA 子集 | knowledge | 30% | 图像实体与外部知识 | 是 |

数据集最终版本以 license、可下载性和标注质量核验为准。缺乏可靠 bbox 的样本仍可训练答案和动作，但 `grounding_eligible=false`，不进入区域 reward 分母。

### 4.2 6,000 条 SFT 轨迹

轨迹来源固定为：

| 类型 | 数量 | 作用 |
|---|---:|---|
| Oracle | 3,600 | 最短正确动作和证据路径 |
| Teacher | 1,500 | 合理的动作与表达多样性 |
| Recovery | 900 | 错误 crop/query 后恢复 |
| 合计 | 6,000 | 其中 5,400 train、600 held-out dev |

SFT 数据检查：

- 轨迹中的答案必须由相应证据支持；
- Teacher 轨迹必须经过规则校验或人工抽检，不能直接相信生成结果；
- Recovery 必须真的恢复，不能把无关步骤插入普通轨迹充数；
- direct-answer、crop、search 三类路径都要出现；
- 每个 `visual_citations/search_citations` 必须引用 ledger 中存在的 id；
- tool result 不参与 action-token loss；
- 对长答案、CoT 和无关叙述做裁剪，训练重点是决策与证据协议。

### 4.3 1,000 个 GRPO 问题单元

主 GRPO manifest：

- perception 500；
- knowledge 500；
- 按 `group_id` 与 SFT dev、sealed test 隔离；
- 每题初始采样 `G=4`，即每个 epoch 最多形成约 4,000 条完整 rollout；
- Stage 1 可先用其中 300–500 题调通；
- Stage 2 使用完整 1,000 题；
- `G=8` 只作为吞吐与方差对照，不预设为最终配置。

### 4.4 800 题 sealed test

- perception 400、knowledge 400；
- 按 image/document/entity `group_id` 隔离；
- 在 SFT/GRPO 前固定 manifest 文件和样本计数，并记录生成时间；
- 训练和开发期间不查看逐题模型结果；
- reward 和 checkpoint 固定后只进行正式评测；
- 如需修 evaluator，只允许在不读取模型答案的情况下修确定性 bug，并保留审计记录。

### 4.5 防止数据泄漏

split 顺序必须是：

```text
先按 group_id 划分 train/dev/grpo/sealed
    → 再生成轨迹、OCR、crop 和反事实版本
```

不得先随机问题再切分，否则同一图片、同一文档页、同一实体或同一模板可能跨集合。至少检查：

- image/document/entity 的来源 id 和去重后的样本计数；
- document id/page id；
- entity id；
- question/answer 近重复；
- search snippet 的答案直接泄漏；
- Teacher prompt 是否包含 sealed 内容。

---

## 5. 环境与系统实验

### 5.1 CROP 固定预算

起始配置：

```yaml
root_visual_tokens: 512
crop_visual_tokens: 256
episode_visual_tokens: 1024
max_crop: 2
crop_source: root_only
```

消融配置使用 `crop_visual_tokens=128`。固定预算应通过 processor 的 `min_pixels/max_pixels` 或等价参数控制，并在处理后再次校验实际 token 数。若超限，继续降低目标分辨率或拒绝动作。

必须覆盖的测试：

- bbox 越界、坐标反转、零面积和极小面积；
- 重复 crop；
- 超大 crop；
- 同 bbox cache hit；
- 1 像素边界取整；
- root normalized 坐标与像素坐标双向转换；
- crop visual token 和 episode token 超限；
- 两条并发 trajectory 的 cache 不串样本。

### 5.2 混合缓存

比较三种系统配置：

| 配置 | root | crop | 用途 |
|---|---|---|---|
| Online-all | 每轮重编码 | 在线编码 | 正确性/速度基线 |
| Hybrid | trajectory 内复用 | 在线编码 | 主线 |
| Root-only | 只编码 root | 无 crop | 无工具系统基线 |

缓存键至少包含：

```text
(image_id, normalized_bbox, target_budget, processor_revision, encoder_revision)
```

主线先只做 trajectory 内缓存。视觉编码器虽然冻结，projector/LLM 仍可能更新，因此跨训练 step 缓存必须在充分验证版本一致性后再考虑。

### 5.3 坐标与 citation sanity test

对每种图片长宽比生成或手标至少 10 个框：

- 正方形；
- 横向长图；
- 纵向文档；
- contact sheet；
- 含 padding/resize 的输入。

验证：

```text
root normalized bbox
    → source pixel bbox
    → crop image
    → ledger evidence_id
    → evaluator canonical bbox
    → root overlay
```

第一版 ANSWER 不再让模型重复输出 bbox。模型引用 `crop_1`，evaluator 从 ledger 读取 root bbox。

### 5.4 Grounding 计算

按 `region_type` 分流：

- `micro`：`Coverage >= 0.8` 或 Gold 中心点在 crop 内，记 `Evidence Hit=1`；
- `macro`：使用 Strict IoU，并报告 `IoU@0.5`；
- 所有区域同时记录 Tightness 或 crop area ratio；
- 引用 root image 可算粗证据，但不给细粒度 bonus；
- 多引用取最能支持答案的证据得分，同时记录引用数量和冗余率。

避免大框刷分：Coverage 命中不等于满分，`grounding_term` 还要结合 Tightness；整页框不能获得与紧凑 crop 相同的区域奖励。

### 5.5 Search 实验

对本地检索分别评估：

- BM25；
- BGE-M3 dense retrieval；
- BM25 + BGE-M3 + RRF；
- 可选 reranker 只在前述组合明显受 precision 限制时启用。

先做离线 retrieval eval，再接入 Agent。报告 Recall@5、MRR、支持证据人工抽检率、单次延迟和索引内存。Agent 评测再报告 Search Call Rate、Search Hit、Search Citation Validity 和重复 query 率。

---

## 6. Baseline 与对照矩阵

### 6.1 主结果必须包含

| ID | 模型/训练 | 工具 | 目的 |
|---|---|---|---|
| B0 | Base 3B | 无，直接回答 | 模型原始能力 |
| B1 | Base 3B + agent prompt | CROP/SEARCH | 排除“只是开放工具”的收益 |
| B2 | SFT 3B | CROP/SEARCH | 冷启动和格式学习 |
| B3 | SFT + Stage 1 GRPO | CROP/SEARCH | 平滑 reward 的工具学习 |
| B4 | SFT + Stage 2 gated GRPO | CROP/SEARCH | 最终系统 |
| O1 | Oracle evidence + Base/SFT | 由环境提供正确证据 | 估计证据质量上界 |
| L1 | Random/center crop | CROP | 排除“任意放大都有效” |

所有主对照固定：

- 同一 3B base revision；
- 同一图片 processor；
- 同一 answer evaluator；
- 同一 visual/context budget；
- 同一 manifest；
- 相同或明确报告的 decoding 参数。

### 6.2 核心消融

| Ablation | 要回答的问题 |
|---|---|
| `w/o CROP` | 主动视觉聚焦是否真正贡献 |
| `w/o SEARCH` | knowledge 流是否依赖检索 |
| `w/o grounding reward` | 区域奖励是否改善可定位证据 |
| `w/o counterfactual aux` | 遮挡训练是否降低视觉捷径 |
| `w/o step cost` | 工具成本是否减少冗余调用 |
| `ungated reward` | 答案门控是否抑制画框刷分 |
| `crop tokens 128 vs 256` | 细节、显存和吞吐的权衡 |
| `online-all vs hybrid` | 缓存的真实系统收益 |
| `bbox re-output vs evidence-id` | canonical citation 是否降低坐标错误 |
| `natural vs 1:1 GRPO sampling` | task balance 是否阻止 Search 退化 |

时间不足时优先级为：`w/o CROP`、`w/o SEARCH`、`w/o grounding reward`、`w/o counterfactual aux`、`online-all vs hybrid`。其余作为第二优先级。

---

## 7. SFT 实验

### 7.1 训练策略

主线初始策略：

- 冻结视觉编码器参数，但每个动态 crop 仍在线前向；
- 训练 LLM/多模态 projector 的范围由 Day 0 显存 profiling 决定；
- 3B 全参数 SFT 为首选，FSDP/ZeRO 不稳定时使用高容量 LoRA 作为明确 fallback；
- tool observation token、padding 和系统模板不计算 action loss；
- action JSON、query、answer 和 citation token 计算监督损失；
- 保存至少一个中期 checkpoint，防止末期过拟合格式。

“冻结视觉编码器”不等于静态 embedding：模型每次执行新 bbox 后仍需在线编码新的 crop，只是不更新 ViT 权重。

### 7.2 SFT 小实验顺序

1. 200 条轨迹 overfit：确认 loss、mask 和保存/加载正确；
2. 1,000 条轨迹 pilot：比较全参数与 LoRA 的显存和 action 合法率；
3. 6,000 条正式 SFT；
4. held-out 600 条 dev 评测；
5. 选择 SFT checkpoint，不查看 sealed test。

### 7.3 SFT 监控

- token-level loss 与 action-only loss；
- action schema 合法率；
- bbox 合法率；
- citation id 有效率；
- tool route accuracy：direct/crop/search；
- answer 指标；
- average tool steps；
- visual token 与 OOM/拒绝动作统计。

如果 answer 提升但 route/citation 不提升，说明监督掩码或轨迹质量存在问题；不能直接进入 GRPO。

---

## 8. Reward 校准与 GRPO

### 8.1 Reward 单元测试

在训练前至少构造以下成对轨迹：

1. 相同正确答案，精确 crop vs 整页 crop；
2. 相同正确答案，有效引用 vs 不存在的 evidence id；
3. 错误答案 + 精确框 vs 正确答案 + 无框；
4. knowledge 正确答案，有 Search 证据 vs 无支持证据；
5. 重复 crop/query vs 最短有效路径；
6. 合法 JSON vs 非法格式；
7. 关键区域遮挡后高置信硬答 vs 降低置信/合理请求证据；
8. 非关键区域遮挡前后答案保持一致。

对每组写明预期 reward 大小关系。若代码结果不符合预期，先修 evaluator，不能靠改学习率补救。

### 8.2 Stage 1：GRPO smoke 与 warm-up

顺序：

1. 32–64 个问题、`G=4`，只验证 rollout→reward→advantage→update；
2. 检查至少一部分 group 内存在 reward 方差；
3. 用 300–500 个问题进入 warm-up；
4. 使用平滑的 AnswerPartial/Evidence/Grounding/Format/StepCost；
5. 每 50–100 step 保存轨迹样本和 reward component 直方图；
6. 观察工具策略，而不是只观察总 reward。

必须监控：

- zero-variance group 比例；
- invalid format/action 比例；
- direct-answer/crop/search 路由分布；
- knowledge Search Call Rate 与 Hit；
- crop area、Coverage、Tightness；
- 重复工具调用率；
- clip fraction、KL、entropy、advantage 分布；
- rollout/learner logprob 差异；
- 单步吞吐、GPU peak、Host RSS。

### 8.3 Stage 2：答案门控 GRPO

使用完整 1,000 问题单元和 1:1 task-balanced sampler，起始 reward：

```text
R_main = I(answer_correct) * (1.0
         + 0.50 * evidence_hit
         + 0.30 * grounding_term
         + 0.20 * citation_validity)
         + 0.20 * counterfactual_aux
         - 0.10 * valid_tool_steps
         - 1.00 * invalid_format
```

实验中记录所有分量，不能只记录加总值。`grounding_term` 按 region type 选择 Coverage/Center-in-Box 或 Strict IoU，并结合 Tightness 抑制大框。

### 8.4 反事实辅助项

- 只对约 15% 可构造样本启用；
- 优先遮挡已标注关键区域；
- 另做非关键区域扰动作为控制组；
- 不把“拒答”本身当正确，需结合原题答案、证据状态和 confidence；
- 记录额外前向成本，比较 10%、15%、20% 采样率后固定一档；
- 全量 sealed 评测报告反事实指标，但不要求所有训练 episode 双前向。

### 8.5 Checkpoint 选择

不用训练 reward 单独选模型。开发集选择分两步：

1. 先排除 answer 明显退化、格式错误或策略坍缩的 checkpoint；
2. 在剩余 checkpoint 中按预注册的 `Joint Success` 选择，并同时报告 Answer 和 Efficiency。

主 run 使用 seed 42；最终方法至少再用一个独立 seed 复现。完整消融可以先用单 seed，关键消融用 bootstrap 或第二 seed 验证不确定性。

---

## 9. 指标与结果表

### 9.1 Answer

- 各 benchmark 官方 Accuracy/ANLS/F1；
- perception、knowledge 分开；
- macro average 与 sample-weighted average 同时报；
- 数值题使用标准化和容差规则，并固化 evaluator 版本。

### 9.2 Evidence 与 Grounding

- Evidence Hit@1；
- Citation Validity；
- Grounding Coverage@0.8（micro）；
- Center-in-Box（micro 辅助）；
- Strict IoU 与 IoU@0.5（macro）；
- Crop Tightness / Crop Area Ratio；
- Joint Success；
- 平均引用数量和冗余 citation 率。

### 9.3 Search

- Search Call Rate（只在 knowledge 分母上计算）；
- Search Hit@k；
- Search Citation Validity；
- Query 重复率；
- 检索后 Answer Acc；
- perception 上的 Search mask violation 必须为 0。

### 9.4 反事实与鲁棒性

- 关键区域遮挡后幻觉率；
- 置信度下降量；
- 合理拒答/请求证据率；
- 非关键区域扰动后的答案保持率；
- 遮挡后恢复 crop/search 的成功率。

### 9.5 策略与效率

- 平均工具步数；
- Crop/Search/Direct route 分布；
- 无效调用和重复调用率；
- 每条 trajectory root/crop/total visual tokens；
- episode latency、rollout throughput；
- GPU peak memory、Host RSS；
- cache hit rate；
- OOM、超预算拒绝和 worker retry 数。

### 9.6 主表模板

| Method | Answer | Joint | Evidence Hit | Coverage/IoU | CF Hallucination ↓ | Tool Steps ↓ | Throughput ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base direct | TBD | TBD | TBD | TBD | TBD | 0 | TBD |
| Base agent | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT + GRPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

所有 `TBD` 必须由保存的逐题结果生成；禁止手填无法追溯的数字。

---

## 10. 十周排期与退出标准

| 周 | 核心任务 | 必须产出 | 退出标准 |
|---|---|---|---|
| 1 | 环境与框架硬闸门 | 两类完整 replay、backend 决策、token/profile | 多轮 logprob/mask/replay 打通 |
| 2 | 数据 adapter、Search、ledger、evaluator | 环境 v0、检索小基线、坐标测试 | 200 episode 稳定运行 |
| 3 | 轨迹生成与清洗 | 首批 Oracle/Teacher/Recovery、manifest | 抽检证据和引用通过 |
| 4 | 6k SFT 与基线 | Base/B1/SFT 结果、SFT checkpoint | schema ≥98%，route/citation 可用 |
| 5 | GRPO smoke、reward 校准 | 32–64 题更新、reward tests | loss/logprob/advantage 正常 |
| 6 | Stage 1 warm-up | 工具策略曲线、无退化 checkpoint | Search/Crop 均有有效信号 |
| 7 | Stage 2 gated GRPO | 最终候选 checkpoint、分量曲线 | Joint 有正向趋势且答案不崩 |
| 8 | 消融与系统 profiling | 核心消融表、缓存/token 对照 | 能解释收益来自哪里 |
| 9 | sealed/cross-domain/鲁棒性 | 主结果、bootstrap/第二 seed、案例 | 测试集未被用于选择 |
| 10 | 固化与简历材料 | README、架构图、demo、复现实验单 | 每个数字可追溯到日志 |

每周五固定进行一次 go/no-go 评审：只允许根据日志决定继续、回退或砍掉次要分支，不能因为某个漂亮案例而跳过系统指标。

---

## 11. 故障处理表

| 故障 | 诊断信号 | 首选处理 | 仍失败时 |
|---|---|---|---|
| 动态多图 vLLM 崩溃 | CUDA assert/shape/position mismatch | full re-prefill | Ray + Transformers |
| rollout/learner logprob 不一致 | KL 异常、训练发散 | 对齐 processor/template/mask | 离线最小序列逐 token 比对 |
| visual token 爆炸 | crop token、GPU peak 突增 | 256→128、拒绝超限 | max crop 2→1、降并发 |
| Host RAM 暴涨 | RSS 随 episode 增长 | 降 worker/prefetch、释放图像对象 | 隔离 rollout worker、磁盘流式读取 |
| Search 坍缩 | knowledge Search Call Rate 接近 0 | 1:1 sampler、检查 reward | 增强 Search SFT/修检索质量 |
| Search 滥用 | 无关 query、重复率高 | step cost、query 去重 | knowledge max_search=1 |
| 大框刷 grounding | Coverage 高但 Tightness 低 | 联合 Tightness | 面积上限/阶梯扣分 |
| 坐标全部偏移 | overlay 统一错位 | canonical remapping | 暂停 grounding reward |
| group reward 零方差 | 大量 group advantage=0 | 增加采样差异/平滑 warm-up | 调整题目难度或 G |
| 只会直接回答 | Crop/Search 调用率低 | 检查 SFT route 与 reward | 分任务 curriculum |
| 只会调用工具 | 步数上升、答案不升 | 增加 direct 轨迹和 step cost | 缩短 max steps |
| 反事实靠拒答刷分 | refusal 上升 | 限制 aux 权重、验证原答案 | CF 仅保留评测 |

---

## 12. 第一个真正要运行的实验：E0 轨迹闭环

### 12.1 目的

E0 不训练模型。它只回答一个决定项目能否继续的问题：

> 同一套环境能否稳定完成、记录并重放 perception 的 Crop 路径和 knowledge 的 Search 路径，同时给出可信的 token、坐标、citation、reward 和系统日志？

### 12.2 样本

- 20 条 perception：10 条需要 crop，5 条可直接回答，5 条含错误 crop 后恢复；
- 20 条 knowledge：15 条需要 search，5 条含错误 query 后恢复；
- 先使用人工/oracle 动作，基础模型输出只用于检查协议，不把模型能力和环境 bug 混在一起。

### 12.3 预期脚本

以下是准备实现的接口名称，不代表当前仓库已经存在或可运行：

```text
scripts/check_env.py
scripts/profile_visual_tokens.py
scripts/validate_manifest.py
scripts/run_trajectory_smoke.py
scripts/replay_trajectory.py
scripts/check_coordinates.py
scripts/summarize_smoke.py
```

### 12.4 E0 产物

```text
artifacts/e0/
├─ manifest.jsonl
├─ trajectories.jsonl
├─ replay_check.json
├─ reward_sanity.json
├─ visual_token_profile.csv
├─ system_profile.json
├─ search_manual_audit.csv
├─ overlays/
└─ report.md
```

### 12.5 E0 通过标准

- 40/40 样本能结构化结束；
- 40/40 轨迹可 replay，ledger 和 evaluator 输入一致；
- 所有 citation id 有效；
- 手工坐标样本 overlay 正确；
- root/crop/total visual tokens 均有记录且不超预算；
- 20 条 knowledge 的 top-5 人工证据命中至少 16 条；
- 200 条压力 replay 无 OOM 或内存持续泄漏；
- 已得到 Online-all 与 Hybrid 的初步延迟和内存数字；
- 所有失败样本进入 `report.md`，不能只保留成功案例。

E0 通过后，下一步才是 200 条 SFT overfit；E0 未通过时，不开始生成 6,000 条轨迹，也不启动 GRPO。

---

## 13. 实验登记清单

每次准备启动正式 run 前复制并填写：

```text
[ ] 研究问题和唯一主变量已写明
[ ] Base/对照使用相同 manifest 和 evaluator
[ ] config 已 resolve 并保存
[ ] model/processor/data/search revision 已固定
[ ] manifest 版本、样本计数已保存且无泄漏
[ ] action/reward/coordinate 单元测试通过
[ ] visual token 与 max context 预算明确
[ ] 预计 GPU/Host RAM 已通过 smoke
[ ] seed 和 checkpoint 选择规则已预注册
[ ] sealed test 未参与调参
[ ] 逐题结果、轨迹和 reward 分量会被保存
[ ] 失败和异常样本也会被保存
```

该计划的原则是：第一周先消灭系统性失败，第 2–4 周建立可信数据和 SFT 基线，第 5–7 周让 GRPO 真正学习工具策略，第 8–10 周用消融、sealed test 和可复现日志证明它不是一个只会演示的 toy。
