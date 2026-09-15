# MM-EvidenceRL 项目方案与十周计划

## 0. 项目定位

**项目名称：** MM-EvidenceRL

**中文名称：** 基于主动视觉证据获取与反事实验证的多模态推理 Agent

**项目性质：** 独立的多模态算法研究与工程项目，不是已有图像恢复项目的改造，也不是简单的 VLM API、VQA Demo 或多模态 RAG 包装。

**目标：** 在十周内构建一个可复现、可评测、可解释的多模态 Agent 训练闭环，使模型在复杂文档、图表和知识密集型视觉问答中学会：

- 判断当前视觉信息是否足够；
- 主动定位和裁剪关键区域；
- 在需要时调用本地文本/视觉检索；
- 保存带来源和坐标的证据链；
- 通过验证动作检查中间结论；
- 在证据不足或关键区域被破坏时降低确信度或拒答；
- 在固定工具和 token 预算下提高回答正确性并减少无效调用。

文档中的研究设想、代码片段、开源项目名称和指标均是讨论材料。实际简历只能填写最终实测结果，不能把规划数字直接当成实验结论。

---

## 1. 研究问题

### 1.1 核心问题

现有开源 VLM 在 Chart、Doc、Math-Vision 和知识密集型 VQA 中常出现三类问题：

1. **视觉捷径和文本先验：** 模型依靠问题文本、常见模式或语言知识猜答案，没有真正读取图像中的关键区域。
2. **视觉推理链缺乏可验证依据：** 模型可以生成流畅的 CoT，但不能指出具体页码、区域、表格单元格或图表元素。
3. **视觉工具使用缺乏策略：** 模型不知道何时放大、何时搜索、何时停止，容易重复裁剪、盲目搜索或在证据不足时继续编造。

本项目研究：

> 在固定的多模态工具环境和预算约束下，能否通过 Visual-Grounded CoT、反事实训练和 on-policy GRPO，使 VLM 学会主动获取、验证并引用视觉证据，从而减少视觉幻觉并提高复杂问答性能？

### 1.2 预期贡献

项目最终应形成四个可被面试官追问、且能够用实验支撑的贡献点：

1. **多模态证据环境：** 将图像、页面、OCR 框、裁剪区域、文本检索结果和证据记忆统一成可交互环境。
2. **结构化视觉动作空间：** 支持 `Crop`、`Ground`、`Search`、`Read`、`Verify` 和 `Answer`，让模型决定下一步如何获取证据。
3. **反事实引导的多目标 GRPO：** 同时优化答案正确性、证据链、视觉 grounding、反事实一致性和工具成本。
4. **面向受限主机内存的训练工程：** 使用异构双节点、分片流式读取和在线 crop 编码，在 8×48GB GPU 与 100GB Host RAM 条件下稳定完成训练。

### 1.3 研究边界

项目不追求以下目标：

- 从零预训练新的视觉语言模型；
- 训练 32B/72B 级别基础模型；
- 一开始端到端更新全部 Vision Encoder；
- 依赖公网搜索、Google Lens 或商业 API 才能运行；
- 只做 Demo，不做固定数据划分、基线、消融和 sealed test。

---

## 2. 总体系统架构

```text
图像 / 多页 PDF / 图表 + 问题
              │
              ▼
     VLM Policy（3B 主线，7B 扩展）
              │
              ▼
      Structured Action Parser
              │
   ┌──────────┼──────────────────┐
   ▼          ▼                  ▼
Visual Crop  Ground          Text Search
   │          │                  │
   └──────────┼──────────────────┘
              ▼
     多模态证据环境与 Evidence Memory
              │
              ▼
          Read / Verify
              │
              ▼
       Answer + Evidence IDs
              │
              ▼
    Rule / Retrieval / Counterfactual Reward
              │
              ▼
        SFT → Multi-turn GRPO
```

### 2.1 环境状态

每个 episode 的 observation 至少包含：

```json
{
  "question": "What is the value of ...?",
  "images": [
    {
      "image_id": "doc_003_page_02",
      "width": 2048,
      "height": 1536,
      "modality": "page"
    }
  ],
  "evidence_memory": [],
  "history": [],
  "budget": {
    "search_left": 4,
    "read_or_crop_left": 6,
    "token_left": 4096
  }
}
```

### 2.2 动作空间

建议在模型输出中使用严格 JSON 或稳定的 XML-like tool tag，并由 parser 做 schema 校验。初始动作集合：

```text
INTERNAL_RECALL
VISUAL_CROP(image_id, bbox, scale)
GROUND(image_id, bbox, label)
TEXT_SEARCH(query)
READ_EVIDENCE(evidence_id)
VERIFY(target_evidence_ids, claim)
ANSWER(answer, citation_ids, confidence)
```

其中 `bbox` 统一归一化到 `[0, 1]` 坐标，避免不同分辨率造成训练和评测不一致。

示例：

```json
{
  "action": "VISUAL_CROP",
  "image_id": "doc_003_page_02",
  "bbox": [0.42, 0.18, 0.87, 0.56],
  "scale": 2.0,
  "reason": "The table header is too small to read reliably"
}
```

### 2.3 Evidence Memory

每条证据必须携带可追溯来源，而不是只保存一段纯文本：

```json
{
  "evidence_id": "doc_003_page_02_box_017",
  "source_type": "visual_crop",
  "image_id": "doc_003_page_02",
  "page_id": 2,
  "bbox": [0.42, 0.18, 0.87, 0.56],
  "ocr_text": "Revenue increased by 18% ...",
  "retrieval_score": 0.81,
  "hop_id": 1,
  "parent_evidence_id": null
}
```

Evidence Memory 需要支持：

- 固定容量，防止上下文无限增长；
- 按图片、页码、bbox 和文本 hash 去重；
- 保留 evidence provenance 和 hop 关系；
- 将当前轨迹中的关键证据注入下一轮 prompt；
- 在最终答案中引用 evidence ID，而非模糊地说“根据图片”。

---

## 3. 数据集与数据管线

### 3.1 任务组成

采用一个统一环境和多个数据适配器，而不是为每个 benchmark 写一套独立逻辑。

| 任务族 | 建议数据集 | 主要能力 | 角色 |
|---|---|---|---|
| 文档理解 | DocVQA、InfoVQA | 版面、表格、局部文字、页内定位 | 主训练任务 |
| 图表与计算 | ChartQA、MathVista | 图表元素、数值读取、多步计算 | 主训练任务 |
| 知识密集型 VQA | InfoSeek、FVQA | 视觉实体识别、文本检索、多跳证据 | 检索子集与迁移测试 |

建议的任务比例：

- 训练混合数据中，Doc/Chart/Math-Vision 占主要部分；
- InfoSeek/FVQA 用于检验是否能把视觉证据和外部知识连接起来；
- 最终评测保留至少一个训练阶段没有见过的来源或版式。

### 3.2 轨迹类型

目标构建约 8,000–12,000 道问题及其候选轨迹。轨迹可以按以下比例生成，实际比例以质量分析为准：

- **50%–60% Oracle 轨迹：** 使用答案和已知证据生成最短可行路径；
- **25%–30% Teacher 轨迹：** 用强教师模型生成多样化搜索、裁剪和验证顺序；
- **15%–20% Recovery 轨迹：** 注入错误搜索、错误 bbox、重复证据或查询漂移，再提供结构化恢复信号。

不要只保留成功轨迹。需要保留一定比例的失败动作，才能训练模型识别无效 crop、无关搜索和证据不足。

### 3.3 反事实数据

对 15%–25% 的样本构造反事实版本：

- 遮挡答案所依赖的关键表格单元格；
- 模糊或裁掉关键图表区域；
- 替换实体、数字或图例；
- 保留问题和非关键背景，破坏真正的视觉证据。

反事实样本的目标不是强迫模型在信息缺失时输出另一个答案，而是检验其置信度和行为是否随证据变化：

- 证据被遮挡且无法恢复时，应拒答或降低置信度；
- 只有非关键区域变化时，答案应保持稳定；
- 关键数字或实体被替换后，模型不应继续机械复制原答案。

### 3.4 Grounding 标注策略

不同数据集的证据框来源不同：

- DocVQA/InfoVQA：使用 OCR word box、layout box 或人工/教师证据框；
- ChartQA：使用图表元素解析框、柱状区域、图例和坐标轴区域；
- MathVista：使用题目图像中的公式、几何对象或关键标注区域；
- InfoSeek/FVQA：保存视觉实体框与检索结果中的支持片段，形成跨模态证据链。

没有可靠 bbox 的样本不能直接用于 IoU 奖励。可以保留用于答案训练，但在 grounding 指标中单独统计。

### 3.5 数据质量门槛

每条训练轨迹进入 SFT 前至少通过：

1. JSON/action schema 合法；
2. bbox 在图像范围内且面积不为零；
3. 引用 evidence ID 确实存在；
4. 证据文本或区域能支持标注答案；
5. 没有连续重复工具调用；
6. 问题、答案、证据来源之间没有 train/test 泄漏。

---

## 4. 模型与视觉编码策略

### 4.1 3B/7B 双阶段策略

不把 3B 视为“缩水版”，而是用它提高 RL 迭代速度，再用 7B 做最终能力和规模对比。

**3B 主线：**

- 用于环境、parser、reward 和 rollout 的快速迭代；
- 可以尝试全参数 SFT；
- 用于完成主要 GRPO 实验和消融；
- 适合在十周内得到多个可比较 checkpoint。

**7B 扩展：**

- 使用同一套数据和环境；
- 优先 LoRA/QLoRA SFT，再做较小规模 GRPO；
- 作为最终增强模型或 scaling ablation；
- 若多模态 rollout 框架不稳定，不牺牲 3B 主线的完整性。

推荐先用一个支持视觉坐标输出、图像裁剪和多图输入的开源 VLM。具体模型和版本必须在第一周锁定并记录 commit/config，不能只在简历中写模型名字。

### 4.2 在线视觉编码

动态 `VISUAL_CROP` 使全局静态 embedding 不再适用。推荐：

1. Vision Encoder 初期冻结，只在线执行前向，不向视觉编码器反传；
2. 原图特征在单条 trajectory 内缓存；
3. 新 bbox 或新分辨率的 crop 重新编码；
4. cache key 使用 `(image_id, normalized_bbox, resolution)`；
5. 训练节点保留原图或可重建的 tile，而不是只传不可逆的全局 embedding；
6. 后期只解冻 projector 或最后几层视觉模块，作为可选实验。

这不是静态特征缓存，而是**轨迹内动态视觉读取缓存**。它保留了 RL 学习“看哪里”的能力，同时避免同一轨迹重复编码同一 crop。

### 4.3 Rollout 架构

优先尝试：

```text
Ray/verl Orchestrator
      ├── rollout workers：在线图像/crop 编码 + 多轮生成
      ├── tool workers：crop、OCR、本地检索、验证
      └── learner workers：Actor/Reference 前向、GRPO 更新
```

建议在第一周验证 `verl + vLLM` 是否支持当前模型的：

- 多轮图像输入；
- 每轮追加新 crop；
- tool response mask；
- actor 权重同步；
- 可变数量视觉 token。

如果现有接口无法稳定处理动态图片，不要为了“看起来用了 vLLM”而硬接。备用方案是写一个 Ray 异步 rollout adapter，使用 Transformers 或模型原生 generation 接口完成同样的 on-policy 采样，并明确记录吞吐差异。

---

## 5. 训练方案

### 5.1 Cold-start SFT

SFT 的目标不是直接追求最终答案分数，而是让模型稳定输出可执行的视觉动作和证据引用格式。

训练内容：

- Observe：识别问题类型和当前可见信息；
- Locate：给出关键区域或候选区域；
- Crop/Ground：调用正确工具并使用规范坐标；
- Search/Read：在需要时检索和读取证据；
- Verify：检查中间结论；
- Answer：输出答案、引用 ID 和置信度。

建议：

- 3B：全参数 SFT 或 LoRA，2–3 epochs；
- 7B：优先 LoRA/QLoRA；
- 对 reasoning/action/evidence/citation token 使用 loss mask；
- 对非法 JSON、无效 bbox 和虚假 citation 增加格式过滤；
- 保存 Base、SFT checkpoint 和数据版本。

### 5.2 Multi-turn GRPO

每个问题采样 4–8 条完整轨迹。每条轨迹允许多轮视觉工具交互，但受固定预算限制：

- `Search` 最多 4 次；
- `Crop/Read/Ground` 合计最多 6 次；
- 推理和工具 token 初始限制在 1.5K–4K；
- 超出预算或非法动作直接结束 episode 并记录原因。

GRPO 训练需要记录：

- group reward 均值和方差；
- 每个 reward component 的分布；
- 组内零方差比例；
- 每类动作的调用率；
- 平均 rollout 长度；
- 重复搜索、无效 crop 和非法格式比例；
- actor/reference KL；
- 训练和 rollout 吞吐。

对于 group 内所有轨迹得到相同 reward 的情况，进行零方差重采样或暂不更新，避免训练信号退化。

### 5.3 两阶段 curriculum

**Stage 1：视觉工具与证据链对齐**

- 正确动作格式；
- bbox 合法性；
- evidence 命中；
- grounding IoU；
- 搜索结果相关性；
- citation validity；
- 重复动作和 token 成本惩罚。

**Stage 2：答案和反事实鲁棒性**

- Answer EM/F1/ANLS；
- 证据链完整性；
- counterfactual consistency；
- 证据不足时的拒答/降置信度；
- 在不增加无谓工具调用的前提下提升答案质量。

不要一开始就让答案 F1 单独主导 reward，否则模型很容易学会猜答案或复制语言先验。

---

## 6. Reward Engine

### 6.1 组合奖励

初始联合分数可以写成：

```text
J = 0.30 * Answer
  + 0.20 * EvidenceChain
  + 0.15 * Grounding
  + 0.15 * Counterfactual
  + 0.10 * CitationValidity
  + 0.10 * Efficiency
```

这只是起始配置，最终应根据 reward 分布和验证集结果调整。每个 component 必须单独记录，不能只保存一个总分。

### 6.2 Answer Accuracy Reward

- Math/Chart：数值、单位、选项和符号使用程序化校验；
- DocVQA/InfoVQA：EM、ANLS、标准化字符串匹配；
- InfoSeek/FVQA：答案匹配与支持证据联合判断；
- 开放回答：使用规则、文本蕴含或独立 evaluator 辅助，但不能完全依赖同一个策略模型自评。

### 6.3 Grounding Reward

对于有标注框或可靠弱标注的样本：

```text
IoU = area(pred_bbox ∩ gold_bbox) / area(pred_bbox ∪ gold_bbox)
```

建议同时统计：

- `Grounding Precision@0.5`；
- `Evidence Hit@1`；
- bbox 合法率；
- 平均 crop 面积；
- 是否命中真正支持答案的区域。

不要只奖励“大框”。需要加入面积、中心偏差或证据支持约束，防止模型输出覆盖整页的伪 grounding。

### 6.4 Counterfactual Reward

对原图和遮挡/扰动图分别运行：

- 关键证据遮挡后仍高度确定且答案完全不变：高额惩罚；
- 关键证据保留、背景变化后答案保持：正向奖励；
- 关键区域破坏后主动请求更多证据或拒答：正向奖励；
- 只改变无关区域却改变答案：稳定性惩罚。

### 6.5 Efficiency 和非法动作惩罚

惩罚项包括：

- 重复 query；
- 重复 crop；
- 不相交或越界 bbox；
- 没有读取结果就引用 evidence；
- 无意义的连续 `Verify`；
- 超出 Search/Read/token budget；
- 过长且没有新增信息的推理链。

惩罚不能过重，否则模型会学会不行动。应先保证正确动作可以获得足够大的正奖励，再逐步增加成本约束。

---

## 7. 评测、基线与消融

### 7.1 必须比较的系统

| 系统 | 作用 |
|---|---|
| Base VLM | 测量模型原始视觉推理能力 |
| Base + Tools | 测量工具本身带来的收益 |
| SFT | 测量格式和冷启动轨迹的贡献 |
| SFT + GRPO | 测量强化学习贡献 |
| GRPO 去掉 Counterfactual | 验证反事实奖励 |
| GRPO 去掉 Grounding | 验证证据定位奖励 |

所有系统必须使用相同测试集、相同工具预算和相同答案后处理。

### 7.2 核心指标

**答案能力：**

- Answer EM；
- Answer F1；
- ANLS；
- Chart/Math 数值准确率。

**证据能力：**

- Evidence Hit@1/Hit@k；
- Evidence Chain All-Hit；
- Grounding Precision@0.5；
- Citation Validity；
- 证据支持率。

**鲁棒性：**

- Counterfactual consistency；
- 关键区域遮挡后的合理拒答率；
- 低分辨率/模糊样本性能；
- 跨数据集和跨版式迁移。

**效率：**

- 平均工具调用次数；
- 无效调用率；
- 重复调用率；
- 平均推理 token；
- 单题 rollout 时间；
- GPU 吞吐、Host RSS、峰值显存。

### 7.3 Sealed Test

最终测试集需要按图片、页面、文档来源或实体切分，避免同一文档的近重复页面同时出现在训练和测试中。

至少报告：

- Base → SFT → GRPO 的完整对比；
- 两个随机种子或 bootstrap 置信区间；
- 主任务与迁移任务；
- 质量和效率是否同时改善；
- 失败案例和 reward hacking 案例。

### 7.4 内部目标，不是预先承诺的结果

可以把以下作为十周项目的验收目标，但只有实测后才能写入简历：

- Answer F1 相对 Base +5–10 个百分点；
- Evidence All-Hit 有明显提升；
- Grounding Precision@0.5 达到可解释的稳定水平；
- 重复/无效工具调用下降 20% 以上；
- 反事实遮挡样本上幻觉率下降；
- 迁移到未见版式时仍优于 Base/SFT。

---

## 8. 硬件与工程方案

### 8.1 双节点分工

#### 2×24GB GPU + 200GB Host RAM

定位为数据与索引中枢：

- 图像解码、resize、dynamic tile；
- OCR、版面分析和图表解析；
- bbox 标注清洗与坐标归一化；
- 反事实 mask/blur/crop 生成；
- 教师模型轨迹生成；
- BM25、BGE-M3 和视觉检索索引；
- WebDataset 分片与质量统计。

#### 8×48GB GPU + 100GB Host RAM

定位为训练和强化学习集群：

- 3B/7B SFT；
- 在线视觉 crop 编码；
- 多轮 rollout；
- GRPO Actor/Reference 更新；
- 正式评测和性能 profiling。

### 8.2 Host 内存控制

100GB 是共享的主机内存，不是每张 GPU 独占 12.5GB。必须控制：

- `num_workers` 从 1–2 起步；
- `prefetch_factor=1–2`；
- 默认 `pin_memory=False`，实测后再开启；
- 使用 WebDataset/Sharded Tar/LMDB 流式读取；
- 不在进程启动时把全部图片或样本加载进 RAM；
- 限制解码后的临时对象生命周期；
- 每个 worker 记录 RSS，出现峰值时定位具体缓存链路。

### 8.3 视觉特征缓存策略

不要把所有动态 crop 预先做成静态 global embedding。建议：

- 原始图像或可重建 tile 以分片形式保存；
- 单条 trajectory 内缓存重复 crop；
- 对高频固定 tile 可做可选离线缓存；
- 保留原始 bbox 和图像版本，确保 evidence 可回溯；
- 训练、评测和 demo 使用相同坐标约定。

### 8.4 GPU 布局

初始可以尝试：

```text
4 GPUs：rollout / online vision encoding
4 GPUs：learner / actor-reference training
```

如果 7B rollout 显存或吞吐不足，再按模型并行度调整为 6+2 或 4+4。不要预先把“4+4”写成已验证结果，需通过实际 profiling 决定。

---

## 9. 十周实施计划

### 第 1 周：锁定研究契约与框架可行性

**任务：**

- 确定主模型版本（3B）和扩展模型（7B）；
- 固定 transformers、verl/vLLM、CUDA、PyTorch 版本；
- 跑通多图输入、动态 crop 和坐标输出；
- 验证多轮追加 crop 是否能完成一次生成；
- 设计 action/evidence schema；
- 建立最小日志格式。

**周末验收：**

- 一条样本可以完成 `observe → crop → read → answer`；
- parser 能拒绝非法动作；
- 能测到单轮视觉编码时间、显存和 Host RSS；
- 若 verl/vLLM 接口不稳定，确定 Ray/Transformers fallback。

### 第 2 周：完成离线多模态环境

**任务：**

- 接入 DocVQA/ChartQA/MathVista 的统一 adapter；
- 完成 OCR/layout/chart evidence 提取；
- 实现 Crop、Ground、Search、Read、Verify；
- 建立固定容量 Evidence Memory；
- 接入本地 BM25+BGE-M3/视觉索引；
- 记录每步 observation、action、tool response。

**周末验收：**

- 不依赖互联网完成一条完整多轮 episode；
- 工具结果可重放；
- 同一 episode 可以动态生成新 crop 并在线编码。

### 第 3 周：完成 Reward 和非 RL Baseline

**任务：**

- 实现答案、evidence、grounding、citation、cost reward；
- 生成反事实遮挡版本；
- 建立 Base、Base + Tools、规则 ReAct baseline；
- 统计工具失败率、bbox 合法率和 evidence 命中率；
- 写第一版 evaluator 和可视化报告。

**周末验收：**

- 随机轨迹的 reward 可以稳定区分“正确证据”和“伪证据”；
- 反事实样本能暴露至少一类视觉 shortcut；
- 无 RL 的工具 Agent 有可复现基线。

### 第 4 周：数据规模化与 SFT 数据冻结

**任务：**

- 构造 8,000–12,000 题的混合数据；
- 生成 Oracle、Teacher 和 Recovery 轨迹；
- 加入反事实 mask、模糊、错误框和干扰证据；
- 执行 schema、bbox、citation、泄漏检查；
- 用 WebDataset/Sharded Tar 打包。

**周末验收：**

- 训练/验证/测试来源隔离；
- 数据抽样人工检查通过；
- SFT 训练集版本冻结并有 manifest。

### 第 5 周：3B Cold-start SFT

**任务：**

- 训练 3B SFT checkpoint；
- 测量动作格式、bbox、引用和答案表现；
- 对比纯答案 SFT 与 action-aware SFT；
- 统计序列长度和视觉 token 分布；
- 完成 7B LoRA SFT 的 smoke test。

**周末验收：**

- action parser 成功率达到可用于 rollout 的水平；
- SFT 相对 Base 在至少一个主任务上有稳定收益；
- 不出现大量无效 crop、重复 search 或虚假 citation。

### 第 6 周：3B GRPO 小规模闭环

**任务：**

- 每题 4 条 rollout；
- 先只启用 Stage 1 tool/evidence reward；
- 开启 group reward、KL、零方差监控；
- 验证 actor 权重更新后 rollout 能同步；
- 跑 100–300 题的小规模 GRPO。

**周末验收：**

- GRPO 不因非法动作导致大量空轨迹；
- reward component 分布可解释；
- action 使用率出现合理变化，而不是全部选择同一个工具。

### 第 7 周：完整 Stage 1 + Stage 2 GRPO

**任务：**

- 加入 Answer、Counterfactual 和 Efficiency reward；
- rollout 扩大到 500–1,000 题；
- 比较 4 条和 8 条 group sampling；
- 对重复动作、长推理和无证据回答做专项分析；
- 保存多个 checkpoint。

**周末验收：**

- SFT → GRPO 在联合指标上有明确趋势；
- 工具调用效率没有因答案提升而严重恶化；
- 能找到并解释至少一个 reward hacking 案例。

### 第 8 周：7B 扩展与系统消融

**任务：**

- 训练 7B LoRA SFT；
- 在相同环境跑小规模 7B GRPO；
- 完成去掉 Grounding、Counterfactual、Cost 的消融；
- 测试不同视觉 crop cache、resolution 和 rollout 数。

**周末验收：**

- 形成 3B/7B、Base/SFT/GRPO 的统一实验表；
- 确认哪个模型作为最终主结果；
- 记录模型大小、吞吐和效果的 trade-off。

### 第 9 周：Sealed Test 与跨域泛化

**任务：**

- 解锁严格隔离的测试集；
- 评测 Doc/Chart/Math/Knowledge 四类能力；
- 做低分辨率、模糊、关键区域遮挡测试；
- 统计 2 个随机种子或 bootstrap 区间；
- 组织失败案例和可视化证据链。

**周末验收：**

- 所有主指标、效率指标和消融完成；
- 能回答“提升来自答案记忆、工具使用还是 grounding”；
- 没有依赖测试集调参。

### 第 10 周：工程固化与简历交付

**任务：**

- 做 GPU/Host memory/rollout throughput profiling；
- 固定最终 checkpoint、数据 manifest 和配置；
- 清理代码，提供一键复现实验脚本；
- 生成系统架构图、reward 曲线、证据链可视化；
- 只把实测数字写入简历。

**最终交付：**

- 可运行的本地多模态 evidence environment；
- 3B 主线和可选 7B checkpoint；
- SFT + multi-turn GRPO 训练脚本；
- reward/evaluator/ablation 代码；
- sealed test 报告；
- README、架构图、失败案例和简历描述。

---

## 10. 风险与备用方案

| 风险 | 影响 | 备用方案 |
|---|---|---|
| 多模态 vLLM/verl 接口不支持动态 crop | rollout 无法稳定运行 | 使用 Ray + Transformers 自定义 rollout adapter |
| bbox 标注不完整 | IoU reward 不稳定 | 采用 OCR/layout 弱标注，缺标样本只参与答案训练 |
| 视觉 token 太多 | 显存和吞吐下降 | dynamic tile、分辨率上限、轨迹内 cache、裁剪预算 |
| GRPO 学会不行动 | 证据链退化 | 先提高有效动作正奖励，再逐步增加 cost penalty |
| GRPO 学会输出大框 | grounding 虚高 | 加面积/中心偏差/证据支持约束 |
| 反事实 reward 被投机 | 模型机械降低置信度 | 同时评估原图正确性和遮挡后合理性 |
| Host RAM OOM | 多卡 worker 崩溃 | 流式 tar、降低 worker/prefetch、关闭 pin memory、监控 RSS |
| 3B 答案能力不足 | 复杂任务上限较低 | 3B 做算法主线，7B 做最终能力 checkpoint |
| 结果波动大 | 简历数字不可信 | 固定 seed、bootstrap/多 seed、报告失败案例 |

### 10.1 硬性止损点

- **第 2 周末仍无法动态 crop：** 先冻结动作空间，使用预生成多尺度 crop，保证环境先跑通；
- **第 5 周 SFT 仍不能稳定输出合法动作：** 暂停 GRPO，先修 schema、数据和 parser；
- **第 7 周 GRPO reward 上升但答案下降：** 回退到 Stage 1，并降低 grounding/cost 权重；
- **第 8 周 7B 框架不稳定：** 以完整 3B 结果作为主线，不让 7B 破坏最终交付。

---

## 11. 简历呈现模板

最终简历应采用“问题—数据—算法—奖励—结果”的结构，不能只写“调用了 Qwen-VL 完成问答”。

```text
MM-EvidenceRL：基于主动视觉证据获取与反事实验证的多模态推理 Agent

研究问题：针对开源 VLM 在复杂文档/图表问答中的视觉捷径、细粒度 grounding 缺失和无效工具调用，
构建支持动态裁剪、区域定位、本地检索与证据验证的多轮多模态推理环境。

数据与环境：混合 DocVQA、InfoVQA、ChartQA、MathVista 与知识密集型视觉问答数据，构建
Oracle/Teacher/Recovery 轨迹及反事实遮挡样本；使用 Evidence Memory 保存图像、页码、bbox、OCR
和检索来源，并以 Sharded WebDataset 解决 100GB Host RAM 下的多卡流式读取。

算法：设计 Visual-Crop/Ground/Search/Verify/Answer 结构化动作空间，先对 VLM 进行 action-aware
Cold-start SFT，再基于在线视觉编码的 multi-turn GRPO 优化答案、证据链、Grounding、反事实一致性
和工具成本。

结果：Base → SFT → GRPO 的 Answer [实测]，Evidence All-Hit [实测]，Grounding Precision@0.5
[实测]，反事实幻觉率 [实测]，平均工具调用 [实测]；报告跨数据集泛化、消融和吞吐/显存指标。
```

简历中的每个数字都要能在实验日志、配置、checkpoint 和评测脚本中复查。

---

## 12. 第一周直接执行清单

1. 建立项目仓库和实验目录：`configs/`、`env/`、`tools/`、`data/`、`reward/`、`train/`、`eval/`。
2. 锁定 3B 基座、Tokenizer、视觉输入格式和坐标规范。
3. 写 `action_schema.json`、`evidence_schema.json` 和 episode 日志格式。
4. 用一张图完成原图输入、crop、重新编码、回答和 citation。
5. 测量单张图和单个 crop 的视觉编码耗时、显存峰值和 Host RSS。
6. 创建一个完全离线的 20–50 题 smoke 数据集。
7. 实现 parser、工具执行器和 rule-based evaluator。
8. 确认 `verl/vLLM` 是否能稳定追加多轮图片；不能则立即切换 fallback adapter。
9. 在没有 RL 的情况下跑通 Base、Base + Tools 和规则 Agent 三个 baseline。
10. 保存第一份可重放 trajectory，作为后续所有训练和调试的回归样本。

---

## 13. 最终判断

十周、8×48GB GPU、另有 2×24GB + 200GB 内存节点，足以支持一个完整而不玩具化的多模态 RL 项目。项目的含金量不依赖把模型堆到更大，而依赖以下闭环是否真正成立：

```text
明确痛点
  → 可交互多模态环境
  → 有来源和坐标的证据链
  → 可验证的组合奖励
  → SFT 冷启动
  → 在线视觉多轮 GRPO
  → 严格基线与消融
  → sealed test 的可复现实测结果
```

3B 可以作为主力研发和 RL 模型，7B 作为最终增强或规模实验；动态 crop 使用在线视觉编码，轨迹内缓存只用于减少重复计算。这样既保留项目的研究野心，又能在十周内形成完整、可信、经得起面试追问的成果。
