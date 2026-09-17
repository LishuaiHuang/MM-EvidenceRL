# ChartQA 与 FVQA：原始数据和 Oracle 轨迹

本文只解释当前正式数据实际使用的两个来源：ChartQA 和 FVQA。重点是把“原始数据长什么样”与“它如何变成一条 Oracle 轨迹”连起来。文中的统计和 JSON 均来自当前落盘文件与生成脚本，不是对完整数据集的泛泛介绍。

## 先建立一个整体概念

原始数据通常只有下面几样东西：

```text
图片 + 问题 + 标准答案
                 └─ ChartQA 另外有图表标注框
                 └─ FVQA 另外有 search-required 标记和离线搜索缓存
```

我们把它转换成环境能够执行和检查的记录：

```text
原始样本
  → 选择任务类型和证据
  → 生成 evidence_id
  → 写入 CROP/SEARCH/ANSWER 动作
  → datasets/manifests/formal_sft_split.jsonl
```

`evidence_id` 是我们在数据准备阶段建立的证据账本编号，不是原始数据自带的字段。模型最终只能引用环境实际返回过的编号。

当前正式数据的规模是：

| 来源 | 任务模式 | 正式数量 | Oracle 数量 | 训练/开发 |
|---|---|---:|---:|---:|
| ChartQA | perception | 4,200 | 4,200 | 3,780 / 420 |
| FVQA | knowledge | 1,800 | 1,800 | 1,620 / 180 |
| 合计 | — | 6,000 | 6,000 | 5,400 / 600 |

当前 6,000 条全部还是 Oracle；Teacher/Recovery 是在这些问题上由计算节点另行生成的派生轨迹。

## 一、ChartQA 原始数据

### 1.1 它是什么、文件长什么样

ChartQA 是“看图表回答问题”的数据。图片一般是柱状图、折线图或其他统计图，问题要求读取局部数值、比较曲线、找年份、做简单计算等。

当前下载包中，训练部分主要有：

- `train_human.json`：7,398 条问答，3,699 个不同图像；
- `train_augmented.json`：20,901 条问答，15,474 个不同图像；
- `png/`：图表图片；
- `annotations/`：与图片同名的结构化图表标注。

一条原始问答很小，形状大致如下：

```json
{
  "imgname": "10095.png",
  "query": "Is the value of Favorable 38 in 2015?",
  "label": "Yes"
}
```

它只告诉我们图片文件名、问题和答案。真正用于定位图表的框在另一个 annotation JSON 中。例如当前样本 `10305.png` 的标注中有：

```json
{
  "general_figure_info": {
    "figure_info": {
      "bbox": {"x": 5, "y": 97, "w": 291, "h": 259}
    },
    "models": [
      {"name": "Approve", "y": ["54", "63"]},
      {"name": "Disapprove", "y": ["31", "26"]}
    ]
  }
}
```

这里的 `bbox` 是左上角加宽高：`x=5, y=97, w=291, h=259`。`models`、坐标轴、图例等字段描述图表内容，但当前第一版只使用 `figure_info.bbox`，不把每一个柱子或文字框都做成工具动作。

本地示例图（数据文件不进入 Git，只在数据节点和 prepared 目录中保存）：

![ChartQA 示例](../datasets/formal/images/chartqa/10305.png)

这张图的原始尺寸是 `309 × 456`。图表主体从大约 `(5,97)` 到 `(296,356)`，标题和脚注在框外。问题是：

```text
What is the data in Jul 2015 ( all two data)?
```

答案是字符串 `"[63, 26]"`，对应图中 Jul 2015 的两条线。

### 1.2 我们怎样从 ChartQA 选出正式样本

生成脚本 [prepare_formal_data.py](../scripts/prepare_formal_data.py) 做了以下事情：

1. 只读取 `train_human.json` 和 `train_augmented.json`；
2. 用 `Path(imgname).stem` 作为 `group_id`，例如 `10305.png` 的组号是 `10305`；
3. 排除 E0 已经使用过的图像组；
4. 同一个图像组只保留一条问答，避免同一张图的多条问题在正式数据中重复出现；
5. 要求图片和同名 annotation 都存在，并且存在 `figure_info.bbox`；
6. 按 human 文件、再 augmented 文件的顺序取满 4,200 条；
7. 将原图复制到 `datasets/formal/images/chartqa/`，将记录写入 manifest。

因此，正式的 4,200 条不是“把 7,398 条人写问题全部放进去”，而是“按图像组去重后选 4,200 张图，每张图配一条问题”。

### 1.3 bbox 如何变成 CROP 参数

原始标注用的是：

```text
(x, y, w, h)
```

环境的 `CROP` 使用根图像像素坐标的：

```text
[x0, y0, x1, y1]
```

所以生成时做：

```text
x0 = x
y0 = y
x1 = x + w
y1 = y + h
```

对 `10305` 而言：

```text
{x: 5, y: 97, w: 291, h: 259}
→ [5, 97, 296, 356]
```

这个框仍然是 root image 坐标，不是裁剪之后的局部坐标。

### 1.4 一条 ChartQA Oracle 轨迹长什么样

对应的正式 manifest 记录会保留问题、答案和框：

```json
{
  "sample_id": "chartqa_10305_00000",
  "dataset": "chartqa",
  "task_mode": "perception",
  "trajectory_type": "oracle",
  "image_path": "datasets/formal/images/chartqa/10305.png",
  "question": "What is the data in Jul 2015 ( all two data)?",
  "answer": "[63, 26]",
  "gold_bboxes_root": [[5, 97, 296, 356]],
  "requires_search": false,
  "answer_evidence_id": "chartqa_10305_00000:crop:0"
}
```

其动作只有两步：

```json
{"action": "CROP", "bbox": [5, 97, 296, 356],
 "evidence_id": "chartqa_10305_00000:crop:0"}
{"action": "ANSWER", "answer": "[63, 26]",
 "evidence_id": "chartqa_10305_00000:crop:0"}
```

把它翻译成人话就是：

```text
1. 我知道图表主体在哪里，先把那里裁出来。
2. 裁剪结果就是证据 crop_0；根据它回答 [63, 26]。
```

`chartqa_10305_00000:crop:0` 是我们生成的证据编号。原始 ChartQA 并没有这个编号，环境在执行 CROP 成功后才建立对应的 evidence ledger 项。

这里的 Oracle 意味着“数据准备者已经知道正确框和标准答案”，所以它不是模型自己探索出来的成功案例，而是用来教模型正确动作格式和最短证据路径的监督轨迹。

## 二、FVQA 原始数据

### 2.1 它和 ChartQA 的区别

FVQA（Fact-based Visual Question Answering）的问题通常先看图识别一个地点、建筑、物体或人物，再回答一个需要外部知识的问题。例如图片本身只显示一座建筑，问题问“它位于哪个城市”。这类问题不能只靠放大图表文字解决，所以在项目里归为 `knowledge`，允许最多一次 `SEARCH`。

当前训练文件是一个 Parquet：

```text
datasets/raw/fvqa/fvqa_train.parquet
行数：4,856
逻辑字段：11 个
```

常用字段的实际形状如下：

| 字段 | 形状 | 含义 |
|---|---|---|
| `prompt` | 消息列表 | 当前通常只有一条 user 消息，正文在 `content` |
| `images` | 图片列表 | 当前每行一张嵌入的图片，内容在 `bytes`，`path` 为 null |
| `reward_model` | 对象 | `candidate_answers`、`ground_truth`、`style` |
| `data_source` | 字符串 | 当前为 `mmsearch_r1/fvqa_train` |
| `data_id` | 字符串 | 如 `fvqa_train_67`，是稳定样本键 |
| `category` | 字符串 | `search_required` 或 `search_free` |

一条原始 Parquet 记录可以还原成下面这样（图片二进制没有展开）：

```json
{
  "prompt": [
    {"content": "Where is this place located?", "role": "user"}
  ],
  "images": [
    {"bytes": "<PNG 二进制>", "path": null}
  ],
  "reward_model": {
    "candidate_answers": "[]",
    "ground_truth": "södertälje",
    "style": "rule"
  },
  "data_source": "mmsearch_r1/fvqa_train",
  "image_urls": null,
  "data_id": "fvqa_train_67",
  "category": "search_required"
}
```

训练文件中有 3,312 条 `search_required` 和 1,544 条 `search_free`。正式数据没有照搬全部数据，而是抽取：

- 1,260 条 `search_required`；
- 540 条 `search_free`；
- 合计 1,800 条。

FVQA 原始行的图片是直接嵌在 Parquet 里的 bytes，不像 ChartQA 那样通过 `imgname` 去找 png。准备脚本把 bytes 写成：

```text
datasets/formal/images/fvqa/<data_id>.png
```

例如 `fvqa_train_67` 的图像尺寸为 `922 × 672`：

![FVQA 搜索样本示例](../datasets/formal/images/fvqa/fvqa_train_67.png)

### 2.2 FVQA 的搜索缓存是什么

FVQA 还带有一个离线缓存：

```text
datasets/raw/fvqa/fvqa_train_image_search_results_cache.pkl
```

它不是文章正文，而是“这个样本过去做图像搜索时返回的结果摘要”。每个缓存项大致是：

```json
{
  "fvqa_train_67": {
    "tool_returned_web_title_list": [
      "Hemstädning i Södertälje | ...",
      "Södertälje city - Wikidata",
      "WAYNE´S COFFEE, Södertälje - ..."
    ],
    "tool_returned_images_urls": [
      "https://...thumbnail-1...",
      "https://...thumbnail-2...",
      "https://...thumbnail-3..."
    ]
  }
}
```

当前缓存覆盖 4,849 个 `data_id`，每个样本通常有 5 个带 rank 的标题/URL 结果。我们把它变成本地 SQLite + FTS5 索引：

```text
datasets/indices/formal_search/evidence.sqlite3
```

主表字段为：

| 字段 | 示例 | 含义 |
|---|---|---|
| `evidence_id` | `fvqa_train_67:search:1` | 我们生成的证据编号 |
| `sample_id` | `fvqa_train_67` | 属于哪个 FVQA 样本 |
| `rank` | `1` | 原缓存中的结果顺序 |
| `title` | `Hemstädning i Södertälje ...` | 搜索结果标题 |
| `url` | `https://encrypted-tbn...` | 缓存中的 URL，可能为空 |
| `source` | `fvqa_search_cache` | 证据来源标记 |

索引目前有 24,046 条 evidence 记录。这里非常重要的一点是：当前索引主要只有标题和 URL，没有统一的网页正文 `snippet`。所以它足以验证“SEARCH 能返回一个可引用的编号”，但不能自动证明标题已经完整支持答案。

### 2.3 搜索 Oracle 是怎么选证据的

对每一个 `search_required` 样本，准备脚本按下面的规则选择一个目标结果：

```text
1. 读取该 data_id 的搜索结果标题，按 rank 从 1 到 5 看。
2. 找到第一个“标题中包含标准答案字符串”的结果。
3. 把它的编号作为 answer_evidence_id。
4. 如果没有标题包含答案，就退回 rank=1。
```

伪代码是：

```python
for rank, title in enumerate(titles, start=1):
    if answer.lower() in title.lower():
        return f"{sample_id}:search:{rank}"
return f"{sample_id}:search:1"
```

在当前 1,260 条正式 `search_required` 样本中：

- 1,001 条选中的标题直接包含答案字符串；
- 259 条没有这样的标题，只能使用 rank=1 的缓存结果。

后 259 条仍可用于协议和动作训练，但其证据支持度必须由 evaluator 单独标记，不能把“rank=1”说成“正文已验证”。

### 2.4 一条 FVQA SEARCH Oracle 轨迹：逐步看

以 `fvqa_train_67` 为例。

原始输入是：

```text
图片：四宫格城市/建筑照片
问题：Where is this place located?
标准答案：södertälje
category：search_required
```

缓存中相关的搜索结果包括：

| rank | evidence_id | title |
|---:|---|---|
| 1 | `fvqa_train_67:search:1` | Hemstädning i Södertälje \| Städhjälp till Bäst Pris \| Hello Clean |
| 2 | `fvqa_train_67:search:2` | Södertälje city - Wikidata |
| 3 | `fvqa_train_67:search:3` | WAYNE´S COFFEE, Södertälje - Restaurant Reviews ... |

因为 rank=1 的标题里已经出现 `Södertälje`，所以 Oracle 选择：

```text
answer_evidence_id = fvqa_train_67:search:1
```

正式 manifest 中的动作目标是：

```json
{
  "action": "SEARCH",
  "query": "Where is this place located?",
  "evidence_ids": ["fvqa_train_67:search:1"]
}
{
  "action": "ANSWER",
  "answer": "södertälje",
  "evidence_id": "fvqa_train_67:search:1"
}
```

把这个过程想成一次“离线可重放的搜索”：

```text
模型/轨迹：我需要查 “Where is this place located?”
        ↓
环境：在本地 SQLite 中查这个样本允许的缓存结果
        ↓
环境返回：evidence_id=...:search:1，title=...Södertälje...
        ↓
模型/轨迹：答案是 södertälje，并引用刚才返回的 evidence_id
```

这里的 `evidence_ids` 是 Oracle 记录的“期望命中结果”，方便监督和回放；在线交互时，模型不应该事先凭空写出一个未知编号。更严格的运行协议是模型只发送 query：

```text
<SEARCH>{"query":"Where is this place located?"}</SEARCH>
```

环境检索并返回：

```text
<SEARCH_RESULT>{
  "evidence_id": "fvqa_train_67:search:1",
  "title": "Hemstädning i Södertälje | ...",
  "rank": 1
}</SEARCH_RESULT>
```

然后模型才能发送：

```text
<ANSWER>{
  "answer":"södertälje",
  "evidence_id":"fvqa_train_67:search:1"
}</ANSWER>
```

因此，`SEARCH` 的作用不是让模型“自己下载网页”，而是让模型提出查询，由本地证据环境返回该样本对应的缓存结果。当前第一版返回的是标题型证据，不是完整网页阅读器。

这里还要区分“Oracle 记录的目标”与“当前在线搜索函数的实际行为”。准备脚本按“答案是否出现在标题中”选择目标 `evidence_id`；而计算节点的搜索函数会把模型提交的 query 分词，在同一个 `sample_id` 的标题上做 FTS 匹配，再返回匹配结果中 rank 最靠前的一条。因此，原始问题不一定能命中 Oracle 选中的标题。例如 `fvqa_train_67` 的问题词是 `where / this / place / located`，其当前缓存标题不含这些词，直接用原问题查询时会得到空结果。这个例子说明：当前 Oracle manifest 适合表达答案与证据的监督目标，但在正式训练前仍需把 query→evidence 的可执行性逐条验收（例如提供更合适的 Oracle query、补充 snippet/别名索引，或明确将该样本标为弱证据），不能把“标题中出现答案”误认为“任意原问题搜索都能成功”。

### 2.5 一条 FVQA search-free Oracle 轨迹

`search_free` 样本不要求外部检索，答案可以直接从图片中识别。例如 `fvqa_train_5`：

```text
问题：Which city or region does this building locate in?
图片：孟买的一座清真寺
标准答案：Mumbai
category：search_free
```

这类样本没有搜索 evidence，准备脚本使用一个表示根图像的内部编号：

```json
{
  "action": "ANSWER",
  "answer": "Mumbai",
  "evidence_id": "fvqa_train_5:root_image"
}
```

`fvqa_train_5:root_image` 也不是原始 FVQA 字段，而是告诉环境：“答案依据的是输入根图像本身”。

## 三、两个数据集放在一起对比

| 项目 | ChartQA | FVQA |
|---|---|---|
| 主要问题 | 图表读取、比较、计算 | 图像实体 + 外部事实 |
| 当前模式 | `perception` | `knowledge` |
| 原始图片 | ZIP 中的 png 文件 | Parquet 中的嵌入 bytes |
| 原始答案字段 | `label` | `reward_model.ground_truth` |
| 定位信息 | annotation 中的 figure bbox | 当前没有统一 root bbox |
| Oracle 动作 | `CROP → ANSWER` | `SEARCH → ANSWER` 或直接 `ANSWER` |
| 证据编号 | `sample_id:crop:0` | `sample_id:search:rank` 或 `sample_id:root_image` |
| 搜索正文 | 不需要 | 当前没有统一 snippet，主要是标题/URL |

## 四、当前实现的几个边界

1. **Oracle 是自动构造，不是 6,000 条人工逐条标注。** ChartQA 框来自已有 annotation，答案来自原始 `label`；FVQA 答案来自 `ground_truth`，搜索证据来自已有 cache。我们做了文件、路径、动作和 evidence 引用审计，但没有声称每条搜索标题都经过人工事实核验。
2. **正式 manifest 当前全是 Oracle。** 目标比例是约 60% Oracle、25% Teacher、15% Recovery；后两类必须在真实环境中生成并通过答案、动作、bbox/query、恢复和 citation 检查。
3. **坐标约定要注意。** 当前正式 Oracle manifest 的 ChartQA bbox 是 root image 的绝对像素坐标；计算节点 Teacher/Recovery 生成器的模型 prompt 使用 0 到 1 的归一化 bbox，再由环境换算成像素。合并两类轨迹时不能直接混用这两个表示。
4. **SEARCH 字段有离线/在线两种形状。** 当前 Oracle manifest 的 SEARCH 记录 `evidence_ids` 列表，表示期望命中；在线交互时，模型不应预先生成 evidence id，而应先提交 query，等环境返回实际的 `evidence_id`。合并成 SFT 数据时应统一成项目最终 action schema。
5. **标题命中不等于完整证据。** 当前 1,260 条 search-required 中有 259 条只能回退到 rank=1；SQLite 也没有统一正文 snippet。这些样本可以先做动作协议 smoke，但正式 evidence-support reward 前应补充可审计的短文本证据。

## 五、可以怎样检查一条样本

拿到 manifest 中的一条记录，可以按下面顺序理解：

```text
1. image_path：先找到根图片。
2. question / answer：知道模型要回答什么、标准答案是什么。
3. task_mode / requires_search：决定能否调用 SEARCH。
4. gold_bboxes_root：perception 样本的正确 root 框。
5. search_evidence_ids：knowledge 搜索样本允许引用哪些本地证据。
6. trajectory：Oracle 已知的最短动作顺序。
7. answer_evidence_id：最后 ANSWER 必须引用的证据。
```

只要记住一件事即可：

```text
ChartQA 的 Oracle 是“知道图表框，所以先 CROP”；
FVQA 的 SEARCH Oracle 是“知道该样本的缓存结果，所以先 SEARCH，再引用环境返回的 evidence_id”。
```
