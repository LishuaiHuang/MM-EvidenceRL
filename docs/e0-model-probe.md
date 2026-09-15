# E0 模型与最小轨迹探针

记录时间：2026-09-15（Asia/Shanghai）  
状态：`OBSERVED`，尚未作为正式训练结果

## 配置

- 环境：`reflectagent-grpo`
- 模型：`Qwen/Qwen2.5-VL-3B-Instruct@main`
- GPU：单卡 `CUDA_VISIBLE_DEVICES=0`
- 图片：仓库已有 `_pdf_qa/page-1.png`，尺寸 `1240 x 1754`
- Processor：显式 `use_fast=True`，与 vLLM 实际路径统一
- 权重缓存：`/amax/home/lishuai/.cache/huggingface`

## 单卡加载与生成

命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n reflectagent-grpo \
  python scripts/e0_model_probe.py --image _pdf_qa/page-1.png
```

首次运行下载并加载两个权重分片，模型加载耗时 `588.229 s`（包含下载）。观测结果：

```json
{
  "input_tokens": 2802,
  "generated_tokens": 32,
  "visual_inputs": {"pixel_values": [11088, 1176], "image_grid_thw": [1, 3]},
  "gpu_peak_allocated_mb": 7730.0,
  "gpu_peak_reserved_mb": 8100.0,
  "host_rss_mb": 3296.4
}
```

缓存中的目标模型约 `7.1G`。根盘可用空间从约 `142G` 降至 `133G`；后续必须为当前阶段 shards、日志和一个 checkpoint 单独留出预算。

## 两条最小轨迹

命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n reflectagent-grpo \
  python scripts/e0_trajectory_probe.py --image _pdf_qa/page-1.png
```

观测结果：

```json
{
  "crop_trajectory": {
    "steps": ["CROP", "ANSWER"],
    "crop_bbox_root": [0.1, 0.1, 0.9, 0.9],
    "input_tokens": 1787,
    "visual_tokens": 1750,
    "latency_seconds": 2.407
  },
  "search_trajectory": {
    "steps": ["SEARCH", "ANSWER"],
    "evidence_id": "MM-EvidenceRL_实验计划.md:29",
    "snippet": "- 模型可见动作只有 `CROP`、`SEARCH`、`ANSWER`；",
    "input_tokens": 2859,
    "visual_tokens": 2772,
    "latency_seconds": 1.948,
    "replay_match": true
  },
  "gpu_peak_allocated_mb": 7729.7
}
```

这验证了：root image 可加载、内存中的 root crop 可重新编码、本地文档检索结果可注入下一轮 prompt，且贪心生成在相同输入下可确定性 replay。Search 使用的是仓库真实文档行，不是合成 evidence。

## 尚未通过的项目

- 尚未验证 vLLM serving、多轮 action logprob、response mask 或 Ray/learner 对接。
- 尚未验证 200 条连续 episode 的 OOM、worker desync 和长时间稳定性。
- 当前 crop/search 轨迹是最小运行探针，不是训练数据或正式 evaluator 结果。

## vLLM 单卡 serving 探针

在同一缓存 snapshot、`reflectagent-grpo` 和 `vLLM 0.11.1` 上执行了单图 `LLM.generate`。观测到：

- 架构解析成功：`Qwen2_5_VLForConditionalGeneration`；
- 权重占用约 `7.16 GiB`；
- vLLM engine 初始化约 `87.5 s`，其中包含 compile、KV cache 和 CUDA graph warmup；
- 单图请求成功生成中文文档描述，未发生 OOM。

vLLM 的 Qwen2.5-VL processor 不接受 `use_fast=False`（会提示参数无效并忽略），因此环境冻结为 Transformers 和 vLLM 都使用 fast `Qwen2VLImageProcessor`。该探针也没有验证 action logprob、response mask 或 learner 对接。
