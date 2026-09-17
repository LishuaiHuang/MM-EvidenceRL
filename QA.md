# 训练协议与证据检索 Q&A

更新时间：2026-09-17

## 本轮状态

- 正式 SFT 数据已组装为 `artifacts/sft-full/formal_sft.jsonl`，共 5,400 条：3,880 条 Oracle、1,520 条 Teacher，Recovery 为 0。
- 正式 LoRA SFT 已完成 5,400 步，输出在 `artifacts/sft-lora-corrected/`，TensorBoard 在 `runs/sft-lora-corrected/`，训练日志在 `logs/sft-lora.log`。
- 当前正式轨迹签名只有 `CROP → ANSWER`、`SEARCH → ANSWER`、直接 `ANSWER`；没有同时包含 CROP 和 SEARCH 的正式样本。
- `sft_train` 中感知 3,780 条，知识且需要 SEARCH 1,138 条，知识但直接回答 482 条。这里的“知识类”不是一律搜索。

## 结论先行

当前不建议为了加入 `<think>` 重做整套 SFT。`<think>` 不是环境动作，也不是
`CROP / SEARCH / ANSWER` 闭环的必要条件。建议保留当前可审计的 action-only 协议，
把模型的内部推理交给模型本身或后续 GRPO 的 reward，不把自由文本 reasoning 变成
新的必须标注字段。

正文 snippet 值得补，但应先改本地 evidence index 和 tool result，不必因为新增一个
`snippet` 字段就重做已有 Oracle manifest。当前索引确实主要只有标题：
`prepare_formal_data.py` 从缓存读取 `tool_returned_web_title_list`，写入
`evidence(evidence_id, sample_id, rank, title, url, source)`，没有正文列。

当前正式 Oracle 轨迹是单工具路径。建议 SFT 冷启动继续使用单步 `CROP → ANSWER`
或 `SEARCH → ANSWER`，等环境的多轮状态、mask、logprob 和 reward 先稳定后，在 GRPO
中开放多步。无需重做全部 6,000 条 Oracle；如果希望 SFT 本身教会多步策略，再追加
一小批多轮 Teacher/Recovery 数据即可。

## Q1：项目是不是缺少 `<think>` 动作？

是的，当前协议没有 `<think>`。这是有意的边界，不是遗漏：

- `AGENTS.md` 将模型可见动作限定为 `CROP`、`SEARCH`、终止 `ANSWER`；
- 当前 manifest 的 trajectory 也只保存这些动作；
- response mask 已覆盖 action/answer response，不依赖额外的 reasoning 字段。

因此当前模型仍然可以在生成 action 前进行模型内部推理，只是不会把推理文本写进
可见轨迹。对本项目而言，证据区域、搜索结果和引用 ID 比一段不可验证的自由文本
更容易 replay 和评分。

## Q2：现在加 `<think>` 是否必须重做 SFT？是否必须人工标注？

如果把 `<think>` 变成每条样本都必须输出的监督字段，原则上需要新增或重生成带
`think` 的轨迹，并同步修改 parser、response mask、轨迹校验和 evaluator。原有
action-only 数据不能直接证明 `<think>` 的内容正确。

但不需要人工逐条撰写：可以让教师模型生成 rationale，再用动作合法性、证据引用、
答案一致性和长度规则过滤。不过教师生成的 reasoning 仍可能幻觉、泄漏 gold answer
或变成模板化废话；人工抽检仍然需要。若只想保留 reasoning 能力，可在后续实验中
把 `<think>` 作为可选字段或只在 GRPO 日志中记录，不把它作为 SFT schema 的硬要求。

代价评估：

| 方案 | 数据改动 | 工程改动 | 风险/收益 | 建议 |
|---|---:|---:|---|---|
| 保持 action-only | 无 | 无 | 最稳，立即进入训练 | **采用** |
| 可选 `<think>`，不计入核心指标 | 少量追加 | 低 | 可做分析，不阻塞训练 | 有余力再做 |
| 强制 `<think>` 并监督 | 大量重生成 | 中高 | reasoning 质量难验证，拖慢 SFT | 暂不做 |

## Q3：当前 Search 是不是只有标题匹配，没有正文 snippet？

对正式 FVQA 索引，基本是这样。现有代码把搜索缓存中的标题和 URL 写入 SQLite，
`evidence_fts` 也只索引 `title`、`url`。`choose_evidence()` 还会优先选择标题中
包含答案字符串的结果。这能快速构成可回放的 evidence ID，但不是正文证据，且有
答案字符串泄漏的风险。

E0 的仓库文档搜索另有一条真实文档行作为 snippet；那只是 E0 探针，不代表正式
FVQA 索引已经有正文。

### 正文 snippet 能否加入？

可以，但要区分数据源：

1. 如果上游搜索缓存已经保存摘要/正文字段，只需在 `evidence` 表增加 `snippet`
   列、把该字段加入 FTS，并更新 search tool result；现有 evidence ID 可以保持不变。
2. 如果缓存只有标题和 URL，就不能从当前 Git manifest 凭空恢复正文。需要数据节点
   按来源、license 和离线可用性重新抓取或使用已有本地语料，再重建 index。
3. 新增 snippet 后，已有 Oracle trajectory 中只保存 evidence ID 的部分不必重做；
   但凡是把标题正文直接拼进训练 prompt 的 Teacher/Recovery 轨迹，都应重新生成或
   至少重新 replay，确保引用和答案仍由 snippet 支持。

建议先做 index-only 改造：保留 `evidence_id` 稳定性，增加 `snippet`、`snippet_source`
和长度上限，搜索返回 `title + snippet + evidence_id`。不要把公网实时搜索引入训练
闭环，也不要把“标题包含答案”继续当作正文支持的替代品。

代价评估：中等。若本地缓存已有正文，约是一次索引重建和几条 evaluator/schema
调整；若没有正文，主要成本在数据授权、离线抓取和重新审计，不是模型训练本身。
这是目前最值得优先补的兼容性工作，尤其针对 knowledge Teacher/Recovery。

## Q4：是不是默认只支持一次 CROP、一次 SEARCH？

是，当前实现和正式 Oracle 数据都按单工具路径设计：

- perception：`CROP → ANSWER`，并屏蔽 `SEARCH`；
- knowledge 且需要检索：`SEARCH → ANSWER`，最多一次搜索；
- knowledge 且无需检索：直接 `ANSWER`。

这不是说项目目标永远只有一步。项目方案把多轮 episode 和 multi-turn GRPO 作为后续
目标，但当前计算节点已验证的最小闭环是单工具路径；正式训练集是 5,400 条混合
Oracle/Teacher 单步轨迹，另有 600 条 dev 记录，dev 不参与训练。

## Q5：如果改成多轮，是否必须重做 SFT？

不必须，取决于目标：

- **只想让 GRPO 探索多步：** 保留现有单步 SFT。GRPO rollout 环境扩大为多步，加入
  history、工具预算、重复调用惩罚和多轮 reward；SFT 不需要重做。
- **想让 SFT 直接教会多步路线：** 需要追加多轮 Teacher/Recovery 轨迹，至少覆盖
  `CROP → CROP → ANSWER`、`SEARCH → ANSWER` 后的失败恢复等真实环境路径。无需丢弃
  原有 Oracle，混合训练即可。
- **想改变动作 schema 或证据状态定义：** 需要重新 replay 受影响的轨迹，并更新
  response mask、logprob 对齐、evaluator 和数据审计。

多轮支持的主要成本不在“把一条 JSON 变长”，而在：

- 每轮 observation/history 和动态 crop 图像的 replay；
- 每轮 action token 的 logprob 与 response mask 对齐；
- visual token、显存、RSS 和最大 episode 长度控制；
- evidence ledger 的新增证据、父子关系和重复调用处理；
- GRPO 的 step cost、失败动作和零方差 group 处理。

## 推荐路线与是否值得重头修改

不值得现在重头修改。按下面顺序推进：

1. 保持当前 action-only、单工具 Oracle SFT，先完成冷启动训练和 checkpoint 验收；
2. 数据节点优先补齐正文 snippet（若上游没有正文，则先确认授权和本地语料）；
3. 用少量真实样本 replay `snippet → SEARCH → ANSWER`，确认 citation/evaluator；
4. 在不改变已有 Oracle 的前提下，追加少量多轮 Teacher/Recovery；
5. GRPO 阶段先开放最多 2–3 个工具步，稳定后再扩大预算；
6. `<think>` 仅作为可选研究变量，不进入当前主线数据协议。

最终判断：

| 项目 | 当前是否阻塞 SFT | 是否值得现在重做 |
|---|---:|---:|
| `<think>` | 否 | 否 |
| 正文 snippet | knowledge 证据质量有影响 | 值得做索引增强，不必重做全部 Oracle |
| 多轮 CROP/SEARCH | 阻塞未来 multi-turn GRPO，不阻塞单步 SFT | 不重做全部；后续追加数据 |

## Q6：SFT 是否严格按感知/知识二分类？输入会明确告诉模型类型吗？

数据有 `task_mode` 元数据。当前训练 prompt 没有插入机器可解析的
`<task_mode=...>` 特殊 token，但会用自然语言明确写出当前路线，例如
`For this perception task, use the route CROP then ANSWER`，或
`For this knowledge task, use the route SEARCH then ANSWER`。也就是说，类型/路线对模型
是显式可见的，只是通过普通文本表达，而不是单独的分类头或特殊 token。模型仍会同时
看到图像、问题和目标 action 轨迹。

当前数据的动作空间仍是三选一：`CROP`、`SEARCH`、`ANSWER`。对大多数正式样本，第一步路线是二选一：感知样本走 CROP，需检索知识样本走 SEARCH；另有 482 条知识样本是直接 ANSWER。不是所有知识问题都强制 SEARCH。

## Q7：现在是不是没有既 CROP 又 SEARCH 的 SFT 样本？

是。对 5,400 条正式训练样本逐条检查，动作签名只有：

```text
CROP → ANSWER       3,780
SEARCH → ANSWER     1,138
ANSWER              482
```

同时含有 CROP 和 SEARCH 的样本数为 0。Recovery smoke 中曾有一条 `CROP → CROP → ANSWER`，但它没有进入正式 SFT 集；它只能作为链路演示，不能当作当前覆盖范围。

## Q8：GRPO 能否进行既 CROP 又 SEARCH 的多轮轨迹？

可以，但需要在 GRPO 环境层显式放开组合路线。当前 SFT 数据不包含这种监督，模型不会因为 SFT 自动学会可靠的 `CROP → SEARCH → ANSWER`。GRPO rollout 可以允许每轮从同一动作集合中选择，并维护：当前图像（root 或 crop）、evidence ledger、搜索结果、历史、最大步数和终止条件。每个动作都应产生新的 observation，最终只在引用有效 evidence 时给答案奖励。

建议后续先支持最多 2--3 个工具步，例如 `CROP → SEARCH → ANSWER` 和 `SEARCH → CROP → ANSWER`，再逐步增加预算。需要补的不是 SFT 全量重做，而是多轮环境状态、动作 mask、逐轮 logprob、重复/无效动作惩罚和组合路线 reward。之后可以追加少量多轮 Teacher/Recovery 数据作为 warm start。

## Q9：四条样本直观长什么样？

下面的 JSONL 形状就是训练数据的核心：每行有图像、问题、答案和 trajectory；训练脚本把问题渲染为 prompt，把 trajectory 序列化为 response，只对 response token 计算 loss，prompt token 的 label 为 `-100`。

### 1. 感知 Oracle

```json
{"task_mode":"perception","question":"Is the value of Satisfied dot line 45 in 2017?","trajectory":[{"action":"CROP","bbox":[5,75,302,286],"evidence_id":"chartqa_6356_00058:crop:0"},{"action":"ANSWER","answer":"No","evidence_id":"chartqa_6356_00058:crop:0"}]}
```

模型看到图片和问题，监督输出 CROP 的像素框，再用该 evidence_id 回答。

### 2. 知识 SEARCH Oracle

```json
{"task_mode":"knowledge","question":"Where is this place located?","requires_search":true,"trajectory":[{"action":"SEARCH","query":"Where is this place located?","evidence_ids":["fvqa_train_67:search:1"]},{"action":"ANSWER","answer":"södertälje","evidence_id":"fvqa_train_67:search:1"}]}
```

模型先生成搜索 query，再引用搜索返回的 evidence。正式索引目前主要返回标题。

### 3. 感知 Recovery

这类样本先记录失败动作，再记录恢复动作；下面来自真实 smoke，不是 Oracle：

```json
{"task_mode":"perception","trajectory_type":"recovery","trajectory":[{"action":"CROP","bbox":[0.0,0.0,0.2,0.2],"status":"failed","evidence_id":"chartqa_10430_00002:recovery_bad_crop"},{"action":"CROP","bbox":[0.0118,0.1048,0.7953,0.8405],"status":"ok","evidence_id":"chartqa_10430_00002:teacher_crop:1"},{"action":"ANSWER","answer":"73","evidence_id":"chartqa_10430_00002:teacher_crop:1"}]}
```

Recovery 会教模型读取失败反馈并再次调用工具。当前正式集没有纳入 Recovery。

### 4. 知识 Recovery（未来格式示例）

当前没有通过质量门槛的真实知识 Recovery，因此下面是**说明格式的构造示例，不是训练数据**：

```json
{"task_mode":"knowledge","trajectory_type":"recovery","trajectory":[{"action":"SEARCH","query":"irrelevant query","status":"failed","evidence_id":"fvqa_train_X:bad_search"},{"action":"SEARCH","query":"Where is this place located?","status":"ok","evidence_id":"fvqa_train_X:search:1"},{"action":"ANSWER","answer":"södertälje","evidence_id":"fvqa_train_X:search:1"}]}
```

要把它用于 SFT，必须由真实环境生成并通过 query、evidence、答案一致性审计，不能把这个占位样本写入正式集。

## Q10：四条样本实际模拟一次 SFT 会怎样？

四条样本会被拼成四个独立的单样本 batch，图片和 prompt 每次重新编码。响应序列分别是两行、两行、三行、三行 JSON action。loss mask 只覆盖这些 response 行，因此不会训练模型复述问题或系统 prompt。四条样本混合训练不会改变类别定义；它只让模型看到成功动作、失败反馈后的恢复动作，以及未来可能的 SEARCH 重试形态。

本轮正式训练实际使用了同样的 LoRA + response-only mask 机制，只是规模为 5,400 条、单 GPU、1 epoch。四样本演示在本地实际跑通了 4 steps，首步 loss 为 `1.9286`，输出到临时目录 `/tmp/mm-evidence-sft-four-20260917`，TensorBoard event 输出到 `/tmp/mm-evidence-sft-four-tb-20260917`；它不替代正式 checkpoint。

## Q11：后续能否先训连接器，再训视觉编码器和 LLM？DeepStack 式多层 ViT 特征是否值得？

可以，工程上可拆成三个阶段：

1. 先冻结 ViT 和 LLM，只训练 vision-language connector，让现有视觉 token 进入 LLM 的接口稳定；
2. 再解冻视觉编码器的高层（必要时配合较小学习率），同时训练 connector，保持 LLM 冻结或只开 LoRA；
3. 最后再决定是否解冻 LLM。每一步都应有独立 checkpoint 和显存预算。

DeepStack 式提取 ViT 低/中/高层特征，送入 LLM，理论上可能提升小目标和精确 CROP 定位，因为浅层保留局部边缘，中层保留结构，高层保留语义。但代价也明确：视觉 token 数量、connector 参数和显存/吞吐都会上升；多层特征还需要位置/层级编码，避免 LLM 混淆同一 patch 的不同层表示。当前 CROP 主要是一次根图裁剪，先用现有单层接口完成 GRPO 环境和 reward，再用小规模定位指标比较 DeepStack；不建议现在为它重做 SFT。

判断标准应是 bbox IoU、有效 evidence 引用率、答案准确率和每样本视觉 token/显存，而不是只看训练 loss。若多层特征不能带来明确定位收益，就保留简单 connector。
