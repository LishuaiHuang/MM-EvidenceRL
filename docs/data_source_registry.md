# 数据源登记表（E0 前）

状态说明：

- `CANDIDATE`：技术上适合，但许可证/上游素材仍需完成核验。
- `CONDITIONAL`：已有数据卡或项目页明确限制，满足条件前不得进入主线。
- `BLOCKED`：当前信息不足或明确不适合主线训练。
- `CLEARED`：只有完成来源、许可证、版本、split/group key 和下载范围审计后才能使用；当前没有数据源达到此状态。

| 数据源 | 官方/数据卡来源 | 版本锚点 | 已观察 split/字段 | grounding / Search 价值 | 许可证与风险 | 当前状态 |
|---|---|---|---|---|---|---|
| ChartQA | [官方仓库](https://github.com/vis-nlp/ChartQA)；[HF 数据卡](https://huggingface.co/datasets/ahmed-masry/ChartQA) | HF commit `af8b6f5c08c95085271561c2a3f9d15f2b5a9031` | train/val/test；`imgname/query/label/type/image`；官方仓库说明 full version 含 annotations | 有 chart element/general figure bbox；perception/CROP 首选候选 | HF tag 为 GPL-3.0；官方说明 annotations 可能 noisy，Pew 图表含人工/heuristic 标注；需确认图片与派生标注再分发边界 | `CANDIDATE` |
| MathVista | [官方代码](https://github.com/lupantech/MathVista)；[HF 数据卡](https://huggingface.co/datasets/AI4Math/MathVista) | HF commit `2b6ad69445fbb5695c9b165475e8decdbeb97747` | `testmini` 1000、`test` 5141；`pid/question/image/answer/metadata` | 可做数学/图表/文档评测；数据卡未提供 root bbox | 数据卡明确：新贡献 CC BY-SA 4.0，但图片/原问题来自原作者；可作 test set，禁止作为 training set | `BLOCKED` for SFT/GRPO；`CONDITIONAL` for eval |
| FVQA | [HF 数据卡](https://huggingface.co/datasets/lmms-lab/FVQA)；MMSearch-R1 论文链接见数据卡 | HF commit `bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5` | train 约 5k、test 约 1.8k；`data_id/prompt/images/reward_model/category`；有 search-required/search-free | knowledge/SEARCH 分流候选；数据卡还提供 image-search cache，但不是本地文本证据 | HF tag 为 Apache-2.0；图片来自 Google Image Search，部分来自 InfoSeek train，GPT4o 生成/人工 test；上游图片与缓存的再使用权需单独核验 | `CONDITIONAL` |
| InfoSeek | [官方仓库](https://github.com/open-vision-language/infoseek)；[项目页](https://open-vision-language.github.io/infoseek/) | GitHub `main` commit `4fe7549433dcf0f7ebd825fb12e7dcb6842f04ed` | train/val/test annotation；`data_id/image_id/question/answer/answer_eval/data_split`；另有 KB mapping 与 human set | knowledge/SEARCH 首选候选；Wiki6M 提供本地语料方向，需构建自己的索引 | 仓库代码 license 为 Apache-2.0，但 annotation、OVEN 图片、Google Cloud 文件和 Wiki6M 派生语料的具体数据条款未在本轮完全核清；不能以代码许可证替代数据许可 | `CONDITIONAL` |
| DocVQA / InfoVQA | [DocVQA 项目页](https://www.docvqa.org/) | 待核验 | 计划中使用 OCR/layout 与答案；具体下载包、版本和 split key 待登记 | 文档局部文字、表格、CROP 候选 | 官方下载/竞赛条款、图片来源和 bbox 再分发边界本轮未核清 | `BLOCKED` until license/split audit |

## 1. 当前建议的 E0 组合

暂定只形成候选，不下载：

- perception：优先从 ChartQA full version 选择少量 train/val 样本；按 `imgname` 作为 `group_id`，保留 annotations 中的原始 chart bbox。
- knowledge：优先核验 FVQA 或 InfoSeek 其中一个；必须同时拥有合法的图像/问题/答案、可离线构建的文本证据来源和稳定 split key，才能进入 E0。
- MathVista：不进入 SFT/GRPO 数据；若许可证条件确认，仅作为独立评测候选。
- DocVQA/InfoVQA：许可证和下载条款未清前不下载。

## 2. 下载闸门

任何 `datasets/raw/` 写入前，必须在本表补齐：

1. 官方来源 URL、具体 revision/日期；
2. 数据许可证和上游图片/文本的附加条款；
3. 允许用途（训练、内部评测、再分发）；
4. split 定义和 `group_id` 唯一键；
5. 预计下载体积、落盘目录和清理方案；
6. E0 仅需的文件/样本范围；
7. 是否存在答案泄漏、同图跨 split 或搜索语料直接泄漏。

未满足以上条目时，状态保持 `CONDITIONAL/BLOCKED`，不下载全量资源。
