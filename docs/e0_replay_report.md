# E0 数据回放准备报告

日期：2026-09-15

## 已落盘并校验的源文件

- ChartQA HF revision `af8b6f5c08c95085271561c2a3f9d15f2b5a9031`：`875370872` bytes；ZIP 完整性检查通过。
- FVQA HF revision `bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5`：`1533321684` bytes；Parquet 可读取，包含 4,856 行和 11 个逻辑字段。
- FVQA train image-search cache：`11283308` bytes；仅作为原始缓存保留，未直接写入 E0 证据。

原始文件位于 `datasets/raw/`，不进入 Git。

## E0 manifest

`datasets/manifests/e0.jsonl` 共 40 行：

- 20 条 ChartQA perception，带 root figure bbox，轨迹为 `CROP -> ANSWER`。
- 20 条 FVQA knowledge，其中 15 条标记 `search_required`，5 条标记 `search_free`。
- 15 条搜索轨迹使用 FVQA 答案 oracle 作为离线回放 evidence，证据来源字段明确标注为 `fvqa_ground_truth_oracle`。这用于验证 SEARCH/引用协议，不代表已完成真实检索质量评估。

生成命令：

```text
PYTHONPATH=/tmp/e0deps python3 scripts/prepare_e0.py
```

图片和 `evidence.jsonl` 保存在 `datasets/e0/`，不进入 Git。MathVista 未写入训练或 E0 目录。
