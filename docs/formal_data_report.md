# 正式 SFT 数据与本地 evidence 索引

更新时间：2026-09-15

## 已生成产物

- `datasets/manifests/formal_sft.jsonl`：6,000 条轨迹；4,200 条 ChartQA perception、1,800 条 FVQA knowledge。
- `datasets/manifests/formal_sft_split.jsonl`：同一批轨迹的稳定 split 版本，按 dataset/group_id 分配 90% `sft_train`、10% `sft_dev`。
- `datasets/formal/images/fvqa/`：1,800 张 FVQA 图像，从 Parquet 的嵌入图像字段抽取。
- `datasets/indices/formal_search/evidence.sqlite3`：24,046 条 FVQA 搜索缓存结果，含 SQLite FTS5 全文索引。

原始数据、抽取图片和 SQLite 文件均不进入 Git。

## Manifest 规则

- 排除了 E0 manifest 中已使用的 ChartQA `group_id` 和 FVQA `sample_id`。
- ChartQA 每个图像组只取一条问答，保留 root 坐标系 figure bbox。
- FVQA 按原始 `category` 取 1,260 条 `search_required` 和 540 条 `search_free`。
- `SEARCH` 轨迹最多一次；搜索样本的 `answer_evidence_id` 指向本地索引中的结果记录。1,001/1,260 条搜索样本的选中标题直接包含答案字符串，其余样本保留原始最高排名结果，供 evaluator 单独判断证据支持度。
- 当前 6,000 条均为 `trajectory_type=oracle`：动作路径由原始 bbox、问题和缓存结果确定，不伪造 Teacher 或 Recovery 轨迹。

## Split 审计结果

- ChartQA：3,780 `sft_train` / 420 `sft_dev`。
- FVQA：1,620 `sft_train` / 180 `sft_dev`。
- sample_id 和 `(dataset, group_id)` 均唯一；同一 group 没有跨 split。
- 所有图片路径、搜索 evidence 引用和动作名均通过审计。

## 生成

```text
PYTHONPATH=/tmp/e0deps python3 scripts/prepare_formal_data.py
```

脚本： [scripts/prepare_formal_data.py](../scripts/prepare_formal_data.py)

Split 与审计脚本： [scripts/audit_formal_splits.py](../scripts/audit_formal_splits.py)

## 下一步

Teacher 轨迹需要计算节点的教师模型/rollout 生成；Recovery 轨迹需要在环境中注入错误 bbox 或查询并记录恢复结果。两类轨迹应在 Oracle 数据验收后单独生成，不能用固定模板冒充。FVQA 图片来源和搜索缓存仍保留其上游条款风险，当前产物用于内部实验，不代表获得再分发许可。
