# E0 数据准备计划

状态：`PLAN`。本文件定义最小准备范围，不代表已经下载或生成数据。

## 1. E0 规模

| 任务族 | 数量 | 预期路径 | 当前候选 |
|---|---:|---|---|
| perception | 20 | 10×CROP、5×直接 ANSWER、5×错误 CROP 后恢复 | ChartQA 优先；DocVQA 需许可证核验 |
| knowledge | 20 | 15×SEARCH、5×错误 query 后恢复 | FVQA 或 InfoSeek，需来源与本地语料核验 |

E0 使用人工/oracle 动作回放来验证环境和数据协议，不把基础模型能力与数据/环境 bug 混在一起。

## 2. 最小 manifest

每条样本在进入 E0 前至少需要：

```json
{
  "sample_id": "chartqa_...",
  "dataset": "chartqa",
  "dataset_revision": "...",
  "split": "train",
  "group_id": "source_image_id",
  "task_mode": "perception",
  "image_path": "datasets/raw/...",
  "question": "...",
  "answer": "...",
  "answer_type": "number",
  "region_type": "micro",
  "gold_bboxes_root": [[0.1, 0.2, 0.3, 0.4]],
  "requires_search": false,
  "search_evidence_ids": [],
  "counterfactual_available": false
}
```

knowledge 样本的 `requires_search=true` 时，必须另有可审计的本地 evidence 记录；不能只放答案或搜索 query。

## 3. split 与泄漏规则

先按 `group_id` 划分，再生成任何派生物：

```text
source records
  → group-level train/dev/grpo/sealed assignment
  → image/OCR/bbox extraction
  → crop/negative/counterfactual derivatives
  → E0 manifest
```

E0 本身不参与正式训练；如果后续将其中样本扩展到 SFT/GRPO，必须重新按来源和图片组检查隔离。ChartQA 首选 `imgname`，InfoSeek 首选 `image_id`，FVQA 至少保留 `data_id` 并核对图像标识；不使用随机 question id 代替来源分组。

## 4. 当前阻塞项

- ChartQA：需要确认 GPL-3.0 数据卡、图片和 noisy annotations 的使用/再分发边界后再下载。
- FVQA：需要确认 Google Image Search/InfoSeek 上游图片权利，以及本地文本证据是否可合法建立。
- InfoSeek：需要确认 annotation、OVEN 图片、Wiki6M 派生语料的许可与下载条款。
- DocVQA/InfoVQA：需要补齐官方下载、竞赛条款和 split 文档。
- MathVista：不得作为 SFT/GRPO 训练集；只在评测用途和原始素材条款确认后考虑。

在这些阻塞项解决前，数据服务器不写入全量 `datasets/raw/`，也不生成 6,000 条轨迹。

## 5. E0 通过前的数据侧产物

通过来源审核后，数据节点才生成：

```text
datasets/manifests/e0.jsonl
datasets/e0/images/
datasets/e0/evidence.jsonl
datasets/indices/e0_search/
```

并为每条失败样本保留原因；不以只保留成功案例作为 E0 结果。
