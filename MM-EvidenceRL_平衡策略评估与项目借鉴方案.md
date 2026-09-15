# MM-EvidenceRL 平衡策略评估与项目借鉴方案

## 0. 先给结论

你和 Gemini 讨论出的“黄金平衡版”是一个明显更好的工程落点：它删除了容易造成环境爆炸的多页调度、独立 OCR 工具、显式 `Ground/Read/Verify` 动作和过多 reward 分量，把核心闭环压缩为：

```text
原图/拼接图 + 问题
        ↓
Qwen2.5-VL-3B Policy
        ↓
CROP(bbox) / SEARCH(query)
        ↓
ANSWER(text, bbox_citations)
        ↓
可验证证据与阶梯式奖励
        ↓
SFT → Multi-turn GRPO
```

这个方向不是 toy。它仍然包含：

- 一个可交互的多模态 rollout 环境；
- 动态视觉重采样和在线 crop 编码；
- 本地离线知识检索；
- 带坐标引用的视觉证据链；
- Cold-start Action-aware SFT；
- Multi-turn on-policy GRPO；
- gated reward、反事实遮挡和工具成本约束；
- Base/SFT/GRPO、消融和 sealed test。

但是 Gemini 的版本还有三点需要修正：

1. **不能把 Evidence Memory 完全删掉。** 可以把它从模型显式 observation 中隐藏，保留为环境内部的最小证据账本；否则无法稳定做引用有效性、证据命中和轨迹回放。
2. **不能把 Counterfactual 彻底删出训练设计。** 它可以不作为每条 rollout 都计算的主 reward，而是在少量遮挡 episode 上作为辅助奖励，并作为全量评测指标保留。这样既控制成本，又保留项目的研究辨识度。
3. **“原图占 90% 计算、混合缓存降低 65%”不能预先写死。** 这应当作为需要 profiling 验证的假设；技术上要缓存的是视觉编码器输出，不是未经验证即可跨轮复用的完整 LLM KV cache。

因此，最终建议不是原始“大一统方案”，也不是 Gemini 的极简三工具 Demo，而是：

> **3 个模型可见动作 + 1 个内部证据账本 + 1 个分阶段 reward 体系。**

---

## 1. 对 PDF 中 Gemini “黄金平衡版”的评价

本文判断依据是《多模态实习项目选型指南》末尾对应页 12/17–17/17 的提问和回复。该材料是方案讨论，不是已经验证的实验报告。

### 1.1 做得正确的地方

#### 把多模态环境从“全能模拟器”改成“有限工具沙箱”

原方案同时想处理：PDF 翻页、页面管理、OCR 框、图像裁剪、外部搜索、证据记忆和多轮验证。对于十周项目，这个范围确实过大。

Gemini 建议把初始输入统一为单图或拼接图，只保留 `CROP` 和 `SEARCH` 两个工具，这个判断是正确的。强化学习需要稳定、可重放、低延迟的环境；页码调度本身不是项目的研究贡献。

但“统一成单图”应理解为**统一视觉上下文接口**，而不是粗暴地把所有页面拼成一张超长图片。推荐：

- 单图任务：直接输入原图；
- 多页任务：最多选取 2–4 页生成 contact sheet 或固定 page window；
- 在内部保留 page offset，保证最终 bbox 可以映射回原始页；
- 不实现让模型主动翻页的工具动作。

这样删除了复杂的 PDF 调度，又没有损失证据定位能力。

#### 删除独立 Ground 动作是合理的

`GROUND(image_id, bbox, label)` 和 `CROP(bbox)` 的确存在功能重叠：一个用于定位，一个用于放大查看。对于第一版环境，单独保留 `GROUND` 会增加动作分支和格式错误率。

更合理的做法是：

- `CROP` 是模型主动获取视觉证据的动作；
- `ANSWER` 必须带 `bbox_citations`；
- evaluator 根据最终引用框计算 grounding 指标；
- 不让模型额外执行一次“画框”工具调用。

Grounding 能力没有被删除，只是从显式工具变成答案协议和评测约束。

#### 删除显式 Read/Verify/Recall 动作是合理的

在这个项目规模下：

- `CROP` 的返回结果就是局部视觉读取；
- `SEARCH` 的返回结果就是检索证据；
- `<think>` 中的自检不需要再包装成 `VERIFY` 工具；
- 原图和 crop 的视觉前向本身已经完成了内部 recall。

把这些都做成显式工具，容易让 Agent 在格式层面忙于“调用工具”，却没有真正改善回答。

不过，Verify 不应完全从系统消失。推荐让它成为**回答后的自动评估步骤**：检查引用是否支持答案、遮挡关键区域后模型是否仍然过度自信，而不是让模型再输出一个高风险的自由格式工具。

#### 3B 作为主力模型是合理的

十周项目的瓶颈不是模型参数规模，而是 rollout、奖励、环境和实验迭代速度。3B 适合：

- 全参数或高 rank LoRA SFT；
- 多轮在线视觉 rollout；
- 多次 reward 设计和消融；
- 在同一数据和环境上快速重复实验。

因此主线只做 3B 是合理选择，不会自动让项目变 toy。项目是否有含金量，取决于是否完成了可验证的 Agent/RL 闭环。

可以保留一个极小的 7B smoke test，但不要把 7B 写成主交付，也不要让它消耗主线时间。

#### 混合缓存方向是正确的

“原图特征复用 + crop 在线视觉编码”是最适合当前硬件约束的折中：

- 原图分辨率固定，可在 episode 开始时编码一次；
- crop 的 bbox 是 policy 动态生成的，只能在线编码；
- 同一 trajectory 内重复 crop 可以命中 cache；
- Vision Encoder 初期冻结，缓存不会因参数更新而失效。

这是值得保留的工程亮点。

### 1.2 Gemini 版本仍然过度简化的地方

#### “OCR 不需要”说得太绝对

不应把 OCR 作为模型 observation 中的独立工具，这一点对；但 OCR 仍然有三项重要用途：

1. 生成 DocVQA 的弱 grounding 标注；
2. 建立本地文本检索索引；
3. 判断 crop 或最终引用区域是否包含支持答案的文字。

正确的处理方式是：

```text
OCR = 离线标注/索引/验证组件
OCR ≠ 模型必须主动调用的工具
OCR 文本 ≠ 默认塞进每轮 observation
```

这样不会浪费上下文，也不会失去可验证评测所需的结构化信息。

#### Evidence Memory 不能只剩最终输出

模型不需要看到原方案的 8 个字段，但环境必须保留最小账本，至少包括：

```json
{
  "id": "crop_1",
  "kind": "visual_crop",
  "source": "image_03",
  "bbox": [0.10, 0.20, 0.30, 0.40],
  "text": "optional snippet",
  "parent": "root"
}
```

检索证据只需：

```json
{
  "id": "doc_1",
  "kind": "search",
  "snippet": "...",
  "source": "local_wiki"
}
```

`hop_id`、retrieval score、详细 provenance 等可以放到日志，不必喂给模型。Evidence Memory 的正确定位是：

> **模型不可见或部分可见的 provenance ledger，而不是一个复杂的第二记忆模型。**

#### Counterfactual 不能只剩一个最终案例图

如果只在最后挑 2–3 个遮挡案例做可视化，项目会失去“反事实引导 RL”的可信度。建议做轻量化处理：

- 训练 rollout 中仅抽取 10%–20% 的 episode 做遮挡比较；
- 不把它设计成独立动作；
- 只提供一个小的辅助奖励；
- 在全量验证集上统计遮挡前后幻觉率和置信度变化。

这样成本远低于每条轨迹都做双倍 rollout，但仍然能够验证“模型是否真的依赖视觉证据”。

---

## 2. 最终推荐的平衡版规格

### 2.1 模型可见动作：3+1

模型只直接选择下面三个动作，`ANSWER` 是终止动作：

```text
1. CROP(bbox)
2. SEARCH(query)
3. ANSWER(text, bbox_citations, search_citations, confidence)
```

`INTERNAL_RECALL`、`READ_EVIDENCE`、`GROUND`、`VERIFY` 不作为模型可见工具。

它们分别变为：

| 原概念 | 最终落点 |
|---|---|
| INTERNAL_RECALL | 模型内部思考，不设工具 |
| READ_EVIDENCE | CROP/SEARCH 返回结果的默认行为 |
| GROUND | ANSWER 中的 bbox citation + evaluator |
| VERIFY | 自动 citation checker + 反事实 evaluator |
| Evidence Memory | 环境内部最小 provenance ledger |

### 2.2 统一视觉上下文

环境不提供 PDF 翻页动作，也不把所有外部信息全部塞入 observation。输入接口为：

```json
{
  "visual_context": {
    "context_id": "sample_00031",
    "images": ["root_image"],
    "page_map": null,
    "task_mode": "perception"
  },
  "question": "...",
  "budget": {
    "max_tool_steps": 2,
    "max_search": 0,
    "max_crop": 2,
    "max_tokens": 2048,
    "root_visual_tokens": 512,
    "crop_visual_tokens": 256,
    "total_visual_tokens": 1024
  }
}
```

多页任务只在预处理阶段选择固定窗口或 contact sheet，环境内部保存映射关系。模型不需要学习翻页策略，项目也不会退化成 PDF 阅读器开发。

`task_mode` 不是答案泄漏，而是数据适配器提供的环境元数据：

- `perception`：Doc/Chart/Math 等答案主要由图像证据决定，`SEARCH` 默认被 action mask 禁用；
- `knowledge`：InfoSeek/FVQA 等需要外部知识，开放一次 `SEARCH`，并要求最终答案能引用检索证据。

正式评测仍要报告两类任务的独立结果，不能用混合平均分掩盖 Search 流退化。

### 2.3 CROP 工具

```json
{
  "action": "CROP",
  "bbox": [0.12, 0.28, 0.76, 0.61],
  "scale": 2.0
}
```

环境执行：

1. 校验 bbox 越界、面积过小和重复 crop；
2. 从原图切出局部区域；
3. 将 crop 重采样到固定视觉 token 预算，再通过 Vision Encoder 编码；
4. 将 crop 视觉 token 追加到下一轮上下文，并检查 episode 总视觉 token 上限；
5. 写入内部 evidence ledger；
6. 返回新的 observation。

第一版只允许从 `root_image` 发起 crop，所有 bbox 都使用 root 坐标系。这样可以先消除局部坐标和全局坐标混淆；如果后续支持嵌套 crop，再启用 ledger 中的仿射变换链。

### 2.4 SEARCH 工具

```json
{
  "action": "SEARCH",
  "query": "entity or concept"
}
```

搜索只访问本地离线索引：

- BM25 负责精确词匹配；
- BGE-M3 负责语义召回；
- RRF 或轻量 reranker 合并结果；
- 返回少量 snippet，不返回整个文档；
- query 去重，超过预算直接终止。

这一步借鉴多模态搜索 Agent 的思想，但不依赖 SerpAPI、Google Lens、Jina Reader 或公网稳定性。

### 2.4.1 任务分流与 Search 防退化

DocVQA、ChartQA、MathVista 大多是 self-contained 任务；InfoSeek/FVQA 才是知识密集型任务。若把它们混在同一个 reward 中，并对所有 `SEARCH` 统一收取步数成本，策略会把 Search 概率学成接近 0。这不是理论上的小问题，而是共享 policy 下非常现实的策略退化风险。

采用三层防护：

1. **环境 action mask：** `perception` 样本的 Search 不可用；`knowledge` 样本最多允许一次 Search。这样不会惩罚一个本来不需要检索的任务，也不会让模型在视觉计算题上学会无意义搜索。
2. **Task-balanced GRPO batch：** 原始数据可以保持约 70% perception、30% knowledge，但 GRPO 更新时按任务族 1:1 采样；SFT 可以保留自然分布。这样 Search 在训练中持续获得有效探索信号。
3. **分任务统计：** 分别记录 perception 的 Crop/Answer 指标和 knowledge 的 Search 命中、检索引用和 Answer 指标，禁止只看一个混合总分。

如果后期希望测试统一 Agent，再取消 action mask，改用一个轻量意图路由器或题目来源元数据作为条件输入；这属于泛化实验，不放在第一版主训练链路里。

### 2.5 ANSWER 协议

```json
{
  "action": "ANSWER",
  "text": "The value is 42.",
  "visual_citations": ["crop_1"],
  "search_citations": ["doc_1"],
  "confidence": 0.86
}
```

第一版优先让模型引用 `evidence_id`，而不是在答案中重新输出一组可能错位的局部 bbox。环境根据 ledger 将 `crop_1` 展开成 root 坐标下的 canonical bbox，再计算 Grounding IoU。没有 crop 时可以引用 `root_image`，但不会获得细粒度区域奖励。不要让模型先调用 `GROUND` 再调用 `ANSWER`，这样既冗余又增加探索难度。

---

## 3. 在线编码与混合缓存：可做，但要把技术边界写清楚

### 3.1 正确的缓存对象

初期建议缓存：

- Vision Encoder 输出；
- Projector/Merger 后、送入 LLM 的 visual embeddings；
- 当前 trajectory 内已经编码过的 crop。

不应未经验证就宣称缓存了跨轮完整的 LLM KV cache。追加新的视觉 token 后，文本上下文的 position、image placeholder 和 attention 组织方式可能要求重新执行部分前向。

### 3.2 推荐实现

```text
episode start
  └─ encode(root image) once

policy emits CROP(bbox)
  ├─ cache hit: reuse (image_id, bbox, resolution)
  └─ cache miss: crop image → online Vision Encoder

append crop tokens + tool result
  └─ continue next turn
```

cache key：

```text
(image_id, normalized_bbox, target_resolution, encoder_revision)
```

### 3.3 必须做的对照实验

至少比较：

1. 全在线：原图和 crop 每次都重新编码；
2. 原图缓存 + crop 在线编码；
3. 无 crop：只输入原图。

报告：

- 单题视觉编码时间；
- 单轮 rollout latency；
- 峰值显存；
- Host RSS；
- rollout throughput；
- 答案和 grounding 是否发生变化。

只有测出真实数字后，才能在简历写“视觉编码开销降低 X%”。

---

## 3.4 Grounding 度量：IoU、Coverage 与 Tightness

单独把 `crop_1` 的整张裁剪区域作为预测框，再与微小 Gold Bbox 计算严格 IoU，会系统性低估“裁剪范围包含目标、模型确实看清并答对”的证据价值。例如 Gold 数字只占原图 0.5%，模型裁剪了包含它的 8% 区域，标准 IoU 可能只有约 0.06。

因此不要把所有视觉证据统一压成 `IoU > 0.5` 一个门槛。定义：

```text
Coverage = Area(Crop ∩ Gold) / Area(Gold)
Tightness = Area(Crop ∩ Gold) / Area(Crop)
```

评分规则：

- **微小文字/数字/图表元素：** `Coverage >= 0.8` 或 Gold 中心点落在 Crop 内，判定 `Evidence Hit=1`；同时使用 `Tightness` 或 crop area ratio 抑制“整页大框”。
- **宏观区域/整行表格/整张图表：** 使用严格 IoU，并保留 `IoU@0.5` 指标。
- **仅引用 root image：** 可以作为有效但粗粒度证据，不给细粒度 grounding bonus。
- **多证据引用：** 取支持答案的最佳证据，同时报告引用数量和冗余比例。

建议最终报告：

```text
Evidence Hit@1
Grounding Coverage@0.8
Strict IoU@0.5 (macro regions)
Crop Tightness / Crop Area Ratio
Citation Validity
```

`grounding_term` 在 reward 中按样本的 `region_type` 选择 Coverage 或 Strict IoU；不能让模型靠输出大框获得与精准区域相同的奖励。

---

## 4. Reward：采用“分阶段门控”，而不是一步到位的硬门

### 4.1 为什么 Gemini 的 Gated Reward 方向正确但仍需分阶段

答案错误时清零 grounding/evidence 奖励，能够阻止模型靠画框得分，这是正确的；但如果从训练第一步就使用硬门，模型一旦答案能力不足，所有轨迹都可能得到相同的零分，GRPO 失去探索信号。

因此采用：

```text
Stage 1：SFT 后的工具技能 warm-up
        使用较平滑的 action/evidence/grounding reward

Stage 2：正式 GRPO
        使用答案门控的联合 reward
```

### 4.2 Stage 1：工具技能 reward

在前期小规模 GRPO 或 reward calibration 中使用：

```text
R_warmup = 0.30 * AnswerPartial
         + 0.25 * EvidenceSupport
         + 0.25 * Grounding
         + 0.10 * Format
         - 0.10 * StepCost
```

这里的 `AnswerPartial` 可以是数值接近度、选项是否正确或标准化字符串的部分匹配，不作为最终成绩。

### 4.3 Stage 2：正式 gated reward

建议的起始形式：

```text
R_main = I(answer_correct) * (1.0
         + 0.50 * evidence_hit
         + 0.30 * grounding_term
         + 0.20 * citation_validity)
         + 0.20 * counterfactual_aux
         - 0.10 * valid_tool_steps
         - 1.00 * invalid_format
```

含义：

- 答错时不能通过 grounding 刷出高分；
- 答对但没有证据时拿到基础分；
- 答对且证据命中时获得额外分；
- `grounding_term` 对微小目标使用 Coverage/Center-in-Box，对宏观区域使用严格 IoU；
- 反事实项只作为有限辅助，不压过答案主目标；
- 每走一步有轻微成本，不鼓励无谓调用；
- 非法格式直接惩罚并结束 episode。

权重只是初始配置。最终需要画出每个分量的分布，并通过验证集调节。

### 4.4 Counterfactual 的轻量化策略

正式训练中只对一部分样本启用：

- 10%–20% episode 使用关键区域遮挡；
- 遮挡后仍高度自信且答案不变：负向辅助分；
- 关键区域被遮挡后主动降低置信度或请求 crop/search：正向辅助分；
- 非关键区域变化不应改变正确答案。

全量测试仍然报告：

- 遮挡后幻觉率；
- 置信度变化；
- 合理拒答率；
- 关键区域恢复能力。

### 4.5 避免 Reward Hacking

需要专门监控：

- 输出超大 bbox；
- 直接输出 `ANSWER` 逃避观察；
- 在答案错误时乱画框；
- 通过拒答刷反事实分；
- 重复 crop 同一区域；
- 通过空 query 或通用词刷搜索相关性。

对应防护：

- bbox 面积、中心偏差和支持证据联合评分；
- 记录 no-tool baseline，不能奖励所有问题都 crop；
- 反事实项不单独决定总分；
- query 去重和无效搜索惩罚；
- 对正确答案和证据命中做严格 evaluator。

---

## 5. 数据规模与十周平衡

### 5.1 推荐规模

为了对齐目标简历中的可核验规模，同时控制十周内的数据质量，主线冻结为：

- **6,000 条 SFT 轨迹：** 60% Oracle、25% Teacher、15% Recovery；
- **1,000 个 GRPO 问题单元：** 每题采样 4 条完整 rollout，任务族按 1:1 采样；
- **500–1,000 道严格 sealed test；** 与训练图片、页面和实体来源隔离；
- **约 15% 训练/验证 episode 带反事实版本。**

额外数据只作为扩展实验，不影响主线 manifest。这样既有截图简历中清晰的数字，也避免为了凑规模引入无法验证的低质量合成轨迹。

### 5.2 数据组成

| 数据来源 | 任务族 | 作用 | 原始数据比例 |
|---|---|---|---:|
| ChartQA | perception | 图表局部读取、数值推理 | 30% |
| DocVQA/InfoVQA | perception | 文档布局、局部文字、表格 | 25% |
| MathVista | perception | 视觉计算与符号推理 | 15% |
| InfoSeek/FVQA 子集 | knowledge | 图像实体 + 本地检索 | 30% |

原始数据可以是约 70:30 的 perception:knowledge，但 GRPO sampler 应按任务族做 1:1 task-balanced batching，避免 Search 退化。若知识密集数据暂时不足，宁可重复采样并严格去重，也不要用大量 self-contained 样本把 Search 的学习信号稀释掉。

如果某一数据集缺乏可靠 grounding 标注，它可以参与答案训练，但不要强行参加 IoU reward。知识密集型样本必须单独检查“问题是否真的需要外部知识”，否则 InfoSeek/FVQA 的 Search reward 也会变成伪信号。

### 5.3 轨迹类型

- Oracle：给出最短可行的 crop/search 路径；
- Teacher：提供一定动作顺序多样性；
- Recovery：包含错误 bbox、无关 query 或错误证据，然后恢复到正确答案。

不需要为每条数据构造极长 CoT。重点是让模型学会：

```text
看不清 → CROP
缺知识 → SEARCH
证据足够 → ANSWER + citation
```

---

## 6. 到底借鉴了 DeepMMSearch-R1 和 MMSearch-R1 吗？

### 6.1 结论

**借鉴了，但目前应明确为“借鉴机制和工程范式”，不是直接复现或简单改名。**

如果只按前面原始方案执行，借鉴关系是存在的但不够清晰；如果采用本文平衡版，二者的借鉴边界会更加明确：

- 从 **MMSearch-R1** 借鉴多轮多模态搜索 Agent 的 rollout 组织、工具交互轨迹、SFT → on-policy GRPO 的训练范式；
- 从 **DeepMMSearch-R1** 借鉴主动视觉聚焦、局部 crop/视觉搜索和细粒度区域归因的思想；
- 在此基础上加入本项目自己的离线环境、混合缓存、轻量 evidence ledger、反事实遮挡和 gated reward。

### 6.2 借鉴矩阵

| 组件 | MMSearch-R1 | DeepMMSearch-R1 | MM-EvidenceRL 的处理 |
|---|---|---|---|
| 多轮工具交互 | 借鉴 | 借鉴 | 保留为 rollout 主干 |
| Search 动作 | 借鉴 | 借鉴/吸收 | 改为本地离线 Search |
| 视觉局部搜索 | 部分借鉴 | 核心借鉴 | 实现为 CROP + 在线编码 |
| Grounding/区域归因 | 部分涉及 | 核心借鉴 | 变为 ANSWER bbox citation |
| SFT → GRPO | 借鉴 | 借鉴 | 保留并做分阶段 reward |
| veRL/vLLM 工程接口 | 借鉴 | 可参考 | 先验证，必要时使用 fallback adapter |
| 公网搜索 API | 不借鉴 | 不依赖 | 统一替换为本地索引 |
| 原始 reward 权重 | 不照搬 | 不照搬 | 自己做 gated + counterfactual |
| 原始数据与指标 | 不照搬 | 不照搬 | 使用可复现数据划分和实测结果 |

### 6.3 MMSearch-R1 适合借鉴什么

适合借鉴：

- 多轮 tool-call trajectory 的消息组织；
- 工具结果如何回注下一轮上下文；
- rollout 中如何记录工具调用和 response mask；
- veRL/vLLM 与 learner 的接口分工；
- SFT 后再做多轮 on-policy GRPO 的顺序。

不应直接照搬：

- SerpAPI、Google Lens、Jina Reader 等公网依赖；
- 它的搜索数据、环境假设和 reward 权重；
- 任何未经核对的脚本、版本或论文数字。

你的工程改造应当是：

```text
公网搜索 API
        ↓
本地 BM25 + BGE-M3 + 可选 reranker
```

这样 rollout 可以在内网稳定重放，不会因 HTTP 超时导致 NCCL timeout 或 worker desync。

### 6.4 DeepMMSearch-R1 适合借鉴什么

按你提供的讨论材料，适合借鉴：

- 模型主动判断是否需要视觉局部聚焦；
- 对大图/小字区域执行 crop-search；
- 把视觉定位和检索结合到多轮推理中；
- 将区域级证据作为策略改进的依据。

你的项目不应该照搬成更大的动作系统，而是把它收敛为：

```text
CROP(bbox) → 在线编码局部区域 → 下一轮推理
ANSWER(..., bbox_citations) → 评估区域归因
```

这保留了 DeepMMSearch-R1 的关键研究精神，同时避免让 `Ground`、`Read`、`Verify` 变成互相重叠的显式工具。

### 6.5 开源复用的边界

第 1 周必须核对：

- 论文、官方代码仓库和 commit 是否对应；
- 是否真正支持当前模型和多轮图像追加；
- 代码许可证是否允许修改、发布和简历展示；
- 训练数据和第三方 API 是否有再分发限制；
- 哪些模块属于原项目，哪些是你的改动。

简历可以写“基于多轮多模态搜索 RL 的开源范式进行二次开发”，但不能写成“独立提出了 MMSearch-R1 的全部方法”。你的原创/工程贡献应落在离线环境、紧凑动作、混合缓存、反事实和奖励设计上。

---

## 7. 十周执行方案：保持深度，控制分支

### 第 1–2 周：借鉴框架并打通最小环境

- 核对 MMSearch-R1/DeepMMSearch-R1 的论文、仓库、license 和接口；
- 前三天完成 vLLM/verl 动态多模态闸门测试：单卡、两轮、第二轮追加 crop、可取 action logprob；
- 实现 `CROP`、`SEARCH`、`ANSWER`；
- 建立内部 evidence ledger 和 episode replay；
- 测量全在线与混合缓存的 latency/RSS/显存，并记录 root/crop visual token 数；
- 如果 veRL/vLLM 动态图片接口不稳定，立刻写 Ray/Transformers fallback。

**必须产出：** 一条可重放的 `image → crop → answer+citation` 轨迹，以及一条 `image → search → answer` 轨迹；第 3 天必须明确主 rollout backend，不能把框架兼容性拖到 GRPO 周。

### 第 3–4 周：数据与 Cold-start SFT

- 统一 ChartQA、DocVQA/InfoVQA、MathVista 和小规模 InfoSeek/FVQA adapter；
- 生成 6,000 左右高质量 Oracle/Teacher/Recovery 轨迹；
- 使用 OCR/layout 作为离线 grounding 和验证信息，不直接塞满 observation；
- 对 3B 做全参数 Action-aware SFT，保留 LoRA 备份；
- 检查 action schema、bbox citation 和答案格式。

**必须产出：** Base 与 SFT 的完整基线，held-out parser compliance、答案和 grounding 结果。

### 第 5–6 周：Stage 1 GRPO 和奖励调试

- 每题先采样 4 条轨迹；
- 只启用较平滑的工具/evidence/grounding reward；
- 使用 perception/knowledge 任务族 1:1 的 task-balanced sampler，并分别记录 Search 调用率和命中率；
- 监控 group reward 方差、非法动作和重复调用；
- 验证模型是否开始学会在看不清时 crop、缺知识时 search；
- 处理零方差 group、空轨迹和视觉 token 超预算轨迹。

**必须产出：** 一个不会因格式错误、环境 bug 或全零 reward 而退化的 GRPO smoke checkpoint。

### 第 7 周：正式 gated GRPO + 轻量反事实

- 切换到答案门控 reward；
- 在 10%–20% episode 上加入遮挡辅助项；
- 对 4 条和 8 条 group sampling 做选择；
- 记录 reward hacking、拒答刷分和大框刷分案例；
- 保存多个 checkpoint，按验证集和效率选择。

**必须产出：** Base → SFT → GRPO 的可解释趋势和 reward component 曲线。

### 第 8 周：消融和工程 profiling

至少完成：

- 去掉 CROP；
- 去掉 SEARCH；
- 去掉 grounding citation reward；
- 去掉 counterfactual auxiliary；
- 去掉 step penalty；
- 全在线 vs 混合缓存。

另外增加两项工程消融：固定 crop token 预算 128 vs 256，以及 root 坐标 citation vs evidence-id canonical remapping。测量：平均工具步数、无效调用率、Search 分任务命中率、rollout throughput、GPU 显存、Host RSS、每条轨迹总 visual tokens。

### 第 9 周：Sealed test 与跨域泛化

- 按图片/文档来源隔离测试；
- 测试未见版式、低分辨率、模糊和关键区域遮挡；
- 报告 Answer、Evidence、Grounding、Counterfactual 和 Efficiency；
- 用两个随机种子或 bootstrap 给出不确定性；
- 整理典型成功和失败案例。

### 第 10 周：结果固化与简历化

- 固定最终 checkpoint、数据 manifest 和配置；
- 将原项目借鉴部分与自己的改动写入 README；
- 生成架构图、reward 曲线、证据框可视化；
- 只把日志和 evaluator 能复查的数字写入简历；
- 录制或保存一个离线可重复的 demo。

---

## 8. 最终简历应该突出什么

不要把卖点写成“实现了三个工具”。简历上应该突出闭环和可验证性：

```text
MM-EvidenceRL：基于主动视觉聚焦与反事实验证的多模态推理 Agent

针对轻量级 VLM 在高分辨率图表/文档中的局部视觉信息丢失与文本先验幻觉，
借鉴多轮多模态搜索 RL 范式，构建离线可重放的 CROP/SEARCH/ANSWER 环境；
设计原图视觉特征复用 + crop 在线增量编码的混合推理流，并以 bbox citation
记录可定位证据。

基于约 6,000 条 Oracle/Teacher/Recovery 轨迹进行 Action-aware Cold-start SFT，
随后使用多轮 on-policy GRPO 学习主动 crop 和本地检索策略；提出答案门控的
阶梯式奖励，联合答案正确性、证据命中、Grounding IoU、轻量反事实一致性和
工具步数成本，抑制无效调用与 reward hacking。

在固定工具预算下，比较 Base/SFT/GRPO 及关键消融，报告 Answer、Evidence Hit、
Grounding、遮挡幻觉率、平均工具调用、rollout throughput 和 Host RSS 的实测结果。
```

真正的数字只能在实验后填写。PDF 中的 `11.4%`、`4.6%`、`1.3 次`、`38%`、`27%` 等数字可以作为目标格式，不能提前当作成果。

---

## 9. 四个高风险问题审计与执行修订

下面四个问题不是吹毛求疵，而是会直接影响策略是否学得到、rollout 是否能跑稳、grounding 指标是否可信。它们的解决方式不是继续增加工具，而是在环境边界、采样器和数据协议上加四个硬约束。

### 9.1 数据集与 Search 动作割裂：真实，会导致策略退化

**问题是否存在：是。** DocVQA、ChartQA、MathVista 大多是 self-contained 任务；如果所有样本都开放 `SEARCH`，同时每次工具调用都收取统一步数成本，那么在绝大多数样本上 Search 都是负收益。共享 policy 最终很可能把 Search 概率压到接近 0。

这不是“训练轮数不够”的问题，而是奖励和数据分布给出的最优策略确实偏向不搜索。只在 prompt 里写“需要时再搜索”不能解决它。

**最终处理：**

1. 用 `task_mode=perception/knowledge` 标记任务族；
2. perception 样本默认 action mask 掉 Search，避免在视觉计算题上扣无意义成本；
3. knowledge 样本开放一次 Search，并要求最终答案能引用检索证据；
4. 原始数据可以约 70:30 混合，但 GRPO batch 按任务族 1:1 采样；
5. 分别报告 perception 的 Crop/Answer 指标和 knowledge 的 Search Hit/Citation/Answer 指标。

这里的 `task_mode` 是环境元数据，不是答案泄漏。它表示该 benchmark adapter 是否提供外部知识源。若后期要研究统一 Agent，再增加一个轻量意图路由器作为单独实验，不把路由学习和 GRPO 主线同时引入。

### 9.2 动态 Crop 的视觉 token 炸弹：真实，会导致显存或上下文雪崩

**问题是否存在：是。** Qwen2.5-VL 类模型的视觉 token 数随输入分辨率和 dynamic resolution 策略变化。大 bbox 被放大后，crop 可能比根图产生更多 token；在 `G=4`、多轮 crop 和梯度/缓存并存时，峰值显存会明显上升。500–1000 只是可能出现的量级，不是当前模型版本的固定常数，必须用 processor 输出的实际 grid/token 数测量。

主要直接风险是 GPU/context OOM；Host RAM OOM 更多来自图像解码、DataLoader worker 和 prefetch，不能把两者混为一个原因。

**最终处理：**

```text
root_visual_tokens ≤ ROOT_BUDGET
sum(crop_visual_tokens) ≤ CROP_BUDGET
root_visual_tokens + crop_visual_tokens + text_tokens ≤ CONTEXT_BUDGET
```

建议第一周从以下安全配置开始，再用真实显存曲线调整：

- root 图像视觉 token 上限：512；
- 单个 crop：128 或 256，二选一；
- 单条 trajectory 总视觉 token：先限制为 1024；
- 每条 trajectory 最多 2 次 crop；
- 超限时返回结构化 `CROP_REJECTED(reason=visual_budget)`，禁止把异常交给 CUDA。

固定预算重采样不能简单按 bbox 面积放大到任意分辨率。实际实现应在 processor 后检查 `image_grid_thw` 或等价的视觉 token 数；root 和 crop 在 rollout、learner、评测三处使用同一预算，否则会出现生成和 logprob 计算不一致。

### 9.3 局部/全局坐标混淆：真实，会让 IoU reward 失真

**问题是否存在：是。** 如果模型在 crop 上看到的是局部坐标，却把局部框直接交给 root-image evaluator，预测区域会整体错位，Grounding IoU 接近 0。3B 模型在多轮图像引用中尤其容易混淆坐标系。

这通常不会立刻触发程序崩溃，但会让正确的 crop 得不到奖励，进而错误地把 grounding 行为学掉。

**最终处理：**

- 第一版只允许从 `root_image` 发起 CROP，不允许嵌套 crop；
- CROP 输入统一使用 root 归一化坐标；
- ANSWER 只引用 `evidence_id`，不要求模型再输出第二套局部 bbox；
- ledger 记录 root bbox 和变换信息，evaluator 自动展开成 canonical root bbox；
- 后续若支持嵌套 crop，再使用仿射变换链做 local→parent→root remapping。

最小 ledger：

```json
{
  "id": "crop_1",
  "root_image": "image_03",
  "root_bbox": [0.12, 0.28, 0.76, 0.61],
  "local_bbox": [0.0, 0.0, 1.0, 1.0],
  "transform": "root_to_crop",
  "source": "CROP"
}
```

这样把坐标对齐从模型记忆问题变成环境确定性转换问题，仍然保留区域级 grounding 评测。

### 9.4 veRL/vLLM 多模态多轮兼容性：真实，是第一周必须关闭的基础设施风险

**问题是否存在：是。** 许多开源 RL 示例首先支持纯文本或单轮图文；多轮动态追加 crop 会涉及 image placeholder、视觉位置编码、可变视觉 token、tool response mask 和 generation logprob 对齐。不能假设当前版本会像纯文本一样自动支持。

这里需要修正一个表述：fallback 不一定要“在 PagedAttention 后直接追加新的图像 token”。更稳妥的是每轮保留完整语义上下文，重新构造包含 root image 和已选 crop 的输入并重前向。这样牺牲速度，但避开增量多模态 KV cache 的底层假设。

**前三天闸门测试：**

1. root image 首轮生成 CROP；
2. 执行 crop 并在线编码；
3. 第二轮追加 crop image；
4. 生成 ANSWER；
5. 取到 action token logprob、tool response mask 和完整 trajectory。

通过标准不是“能生成一句话”，而是 generation、logprob、mask、视觉输入和 replay 全部一致。若任一步不稳定，第 3 天立即切换到：

```text
Ray rollout worker
  → Transformers 或稳定的 vLLM full re-prefill
  → 每轮重建完整多模态上下文
  → 只对 policy action tokens 计算 logprob
  → learner 使用同一序列和同一视觉预算更新
```

这会降低 rollout throughput，但不会破坏 on-policy GRPO 的算法闭环。吞吐损失作为工程指标报告，不能为了保住某个框架名而把项目拖到第六周才发现无法训练。

### 9.5 四个问题对应的最小验收矩阵

| 风险 | 第 1–2 周必须测的信号 | 失败时的硬动作 |
|---|---|---|
| Search 退化 | 两类任务 Search 调用率、knowledge Search Hit | 启用 task mask + 1:1 GRPO sampler |
| Token 爆炸 | 每轮/root/crop visual tokens、峰值显存 | 降低 crop budget、拒绝超限动作 |
| 坐标错位 | root/crop citation 可视化、IoU sanity check | 改为 evidence-id citation，禁用嵌套 crop |
| 框架断层 | 两轮动态图片、action logprob、mask、replay | full re-prefill fallback |

---

## 10. 最终判断

这套平衡策略的核心判断是对的：

- 动作少，不等于项目浅；
- Evidence Memory 简化，不等于证据链删除；
- 只做 3B，不等于 toy；
- 在线 crop 编码，不需要放弃原图缓存；
- Gated Reward 比六项线性加权更稳，但必须配合 warm-up；
- 借鉴优秀开源工作，不等于照搬；
- 项目的原创性来自环境改造、奖励约束、反事实验证和完整实验闭环。

最终项目应保持下面这个“最小但完整”的结构：

```text
MMSearch-R1 的多轮搜索/RL 范式
        +
DeepMMSearch-R1 的主动视觉聚焦思想
        +
本项目自己的离线证据环境
        +
混合缓存与在线 crop 编码
        +
答案门控 + 轻量反事实 reward
        +
严格基线、消融和 sealed test
```

这就是十周内最有希望达到截图中“研究问题—数据管线—算法—奖励—实验结果”水平的 balance：

> **不把所有概念都做成独立模块，但每个留下来的模块都能在代码、日志和实验中被验证。**
