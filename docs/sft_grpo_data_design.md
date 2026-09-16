# SFT 与 GRPO 数据设计说明

更新时间：2026-09-16

本文回答两个问题：冷启动 SFT 到底教什么，以及后续 GRPO 的问题单元、轨迹和搜索协议如何构造。文中明确区分当前已经落盘的实现和目标设计。

## 1. SFT 先教什么

SFT 的目标不是让模型直接达到最终答案分数，而是先让模型稳定掌握工具环境的“动作语法”和最短证据路径：

- perception 问题看不清关键区域时调用 `CROP`；
- knowledge 问题缺少外部知识时调用一次 `SEARCH`；
- 证据足够后停止调用工具并执行 `ANSWER`；
- 输出合法 JSON、合法 root 坐标 bbox 和真实 `evidence_id`；
- 将答案与视觉区域或搜索结果绑定，而不是只猜答案。

直接做 RL 的问题是早期策略同时面对模型能力、动作格式、工具环境和 reward 设计四个未知量。大量 rollout 可能只是非法 JSON、无效 bbox 或空搜索，组内 reward 还会出现零方差，RL 没有稳定的学习信号。SFT 先把可执行轨迹分布和协议教给模型，GRPO 再优化“何时行动、行动到哪里、证据是否足够”等策略问题。

## 2. 当前 6,000 条 SFT 的真实状态

当前 [formal_sft_split.jsonl](../datasets/manifests/formal_sft_split.jsonl) 有 6,000 条轨迹：4,200 条 ChartQA、1,800 条 FVQA；5,400 条 `sft_train`、600 条 `sft_dev`。它们全部是 `trajectory_type=oracle`，即当前实际比例是：

| 类型 | 当前数量 | 当前比例 | 目标比例（方案） |
|---|---:|---:|---:|
| Oracle | 6,000 | 100% | 60%（约 3,600） |
| Teacher | 0 | 0% | 25%（约 1,500） |
| Recovery | 0 | 0% | 15%（约 900） |

这些轨迹由原始答案、ChartQA figure bbox、FVQA 图像和搜索缓存自动转换而来，没有新增人工逐条标注。结构审计已完成，但这不等同于人工确认每条 FVQA 搜索结果都能充分支持答案。

## 3. 为什么不能只有 Oracle

Oracle 路径短、干净、容易验证，适合教会模型“正确动作长什么样”。如果 6,000 条全部 Oracle，模型可能学到固定模板：看到 ChartQA 就无条件 `CROP`，看到 FVQA 就固定 query 后 `SEARCH`，但不会处理动作顺序变化、无关证据和工具失败。它还缺少错误后的恢复行为，在线 GRPO 初期会浪费大量采样在格式和环境错误上。

### Oracle

Oracle 是已知正确证据后的最短可行路径。例如：

```text
问题：What's the color of the graph with the 15 as the lowest value?
已知 bbox：[5, 62, 317, 293]

{"action":"CROP","bbox":[5,62,317,293],"evidence_id":"crop_0"}
{"action":"ANSWER","text":"Blue","citations":["crop_0"]}
```

### Teacher

Teacher 是强教师模型在同一环境中生成的合理路径。它不是故意绕路，而是提供动作顺序和 query 表达的多样性：

```text
先判断问题依赖图例
→ CROP 图表主体和图例区域
→ 读取颜色与数值对应关系
→ ANSWER "Blue"，引用 crop_0
```

Teacher 的作用是扩大有效行为分布，减少模型把某个固定 bbox 或固定 query 当成答案捷径。当前尚未生成 Teacher 轨迹。

### Recovery

Recovery 在输入轨迹中保留一次可恢复错误，然后记录正确的后续动作。错误不是监督目标，最终监督目标是“发现证据不足并恢复”：

```text
CROP [0, 0, 0.2, 0.2]       # 只截到标题，证据不足
→ observation: insufficient_evidence
→ CROP [5, 62, 317, 293]    # 重新定位图表区域
→ ANSWER "Blue"，引用 crop_1
```

它训练模型不要把第一次错误结果硬当成最终答案，也不要在无证据时继续编造。当前尚未生成 Recovery 轨迹。

## 4. 三类轨迹是否使用相同样本

建议使用同一批“底层问题池”的配对子集，但不要求三类对每一条样本都重复：

- Oracle、Teacher、Recovery 可以针对同一个问题生成多种轨迹，便于比较路径差异；
- 也可以用不同问题补足目标比例，但必须保持 dataset、task_mode 和难度分布一致；
- 所有派生轨迹共享同一个 `group_id`，因此必须进入同一个 split，不能让 Oracle 在 train、Recovery 在 dev；
- E0 样本不直接混入正式 SFT；正式 train/dev 之间仍按图像组或实体隔离。

实操上可先从当前 6,000 个 Oracle 问题中抽取一部分生成 Teacher/Recovery，再去重后保持总量 6,000。例如 3,600 个 Oracle 问题、1,500 个 Teacher 派生轨迹和 900 个 Recovery 派生轨迹；同一问题的多条轨迹算不同 `trajectory_id`，但不能算不同 `group_id`。

## 5. GRPO 数据如何构造

GRPO 不把“正确轨迹”提前写死，而是准备问题单元，让当前 policy 在线生成多条完整 episode。方案目标是 1,000 个问题单元，每题 `G=4` 条 rollout，perception/knowledge 各 500 个问题单元。问题单元与 SFT 按图像组、页面或实体隔离。

一个 perception 问题单元可以是：

```text
输入：root image + "What's the color of the graph with the 15 as the lowest value?"

Rollout A: CROP 正确图例区域 → ANSWER Blue + crop citation
Rollout B: 直接 ANSWER Blue，但没有 crop citation
Rollout C: CROP 标题区域 → ANSWER 错误
Rollout D: CROP 正确区域 → 重复 CROP → ANSWER Blue
```

reward 分别检查答案、证据命中、citation 合法性、grounding、工具步数和重复动作。A 通常最高，B 可能只有基础答案分，C 较低，D 因重复调用扣除成本。knowledge 单元同样采样 4 条，但 knowledge 最多允许一次 `SEARCH`，perception 环境将 `SEARCH` mask 掉。

GRPO 记录的是每个问题的 4 条实际 rollout、每个 reward 分量和环境结束原因，而不是只保存最高分答案。这样才能统计 Search 调用率、evidence hit、非法动作和组内 reward 方差。

## 6. 是否需要固定 prompt 模板

需要固定的系统 prompt 和 action schema，但不能把正确路径或答案写进 prompt。固定部分用于约束协议，例如：

```text
You are an evidence-grounded visual QA agent.
Available actions: CROP, SEARCH, ANSWER.
Return exactly one JSON action per turn.
CROP uses [x0,y0,x1,y1] in root-image pixel coordinates.
SEARCH is allowed only when the environment exposes it and accepts one query.
ANSWER must include text and citations using evidence_id values returned by tools.
Stop after ANSWER. Do not invent evidence_id values.
```

`task_mode`、action mask、工具预算和 evidence ledger 由环境控制，不能只依赖 prompt。这样模型即使违反格式，也会被 parser/环境明确记录，而不是静默修正。

## 7. `<SEARCH>` 标签里放什么

推荐把模型动作和工具返回分开：模型发出的 `<SEARCH>` 只包含检索 query，不提前填写 evidence id：

```text
<SEARCH>{"query":"location of the pictured place"}</SEARCH>
```

环境返回：

```text
<SEARCH_RESULT>{
  "evidence_id":"fvqa_train_67:search:1",
  "title":"Hemstädning i Södertälje | Städhjälp till Bäst Pris | Hello Clean",
  "source_url":"..."
}</SEARCH_RESULT>
```

之后模型才能：

```text
<ANSWER>{"text":"södertälje","citations":["fvqa_train_67:search:1"]}</ANSWER>
```

当前实现的事实是：FVQA 本地 SQLite 索引保存 `evidence_id`、标题、可用 URL 和 rank，主要检索字段是标题；没有统一的正文 snippet。因而当前不能声称训练时已经返回“完整检索证据”。标题可以用于协议和初步 Search 行为 smoke test，但不足以支撑最终的 evidence-support reward。正式 knowledge SFT/GRPO 前，应补充有许可的正文或短文本片段，并在 `<SEARCH_RESULT>` 中加入 `snippet`；在此之前只能把标题型结果标记为弱证据，不能当作人工确认的事实依据。

## 8. 当前与下一步

当前已完成：6,000 条 Oracle SFT、train/dev split、标题型 FVQA 本地 FTS 索引。

尚未完成：Teacher/Recovery 轨迹、1,000 个 GRPO 问题单元、每题 4 条在线 rollout、正文 snippet 证据和 action logprob/response mask 对接。

因此，下一步顺序应是：先冻结上述 action/prompt/search schema，再由计算节点生成并保存 Teacher/Recovery；数据节点补齐合法文本证据字段和 GRPO 问题单元 manifest，随后才启动正式 GRPO。
