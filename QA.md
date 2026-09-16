# 训练协议与证据检索 Q&A

更新时间：2026-09-16

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
目标，但当前计算节点已验证的最小闭环是单工具路径，正式 6,000 条也全部是 Oracle
单步轨迹。

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
