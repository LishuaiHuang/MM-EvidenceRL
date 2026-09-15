# Data Node Environment Profile

盘点范围：数据节点第一轮启动任务。本文只记录只读检查结果，不包含凭据、环境变量、原始数据或模型文件。

盘点分支：`data/bootstrap`

## 1. 执行的检查

```text
pwd
git status --short
git pull --ff-only
df -h
free -h
nvidia-smi
python3 --version
```

Python 包版本通过 `importlib.metadata` 查询；另外用 Torch 做了 CUDA 可见性检查。过程中未安装、升级或下载任何依赖和数据。

## 2. 观测结果

### 工作区

```text
path: /opt/data/private/lishuai/projects/MM-EvidenceRL
branch: data/bootstrap
git status: clean
git pull --ff-only: Already up to date.
```

### 磁盘

项目所在文件系统为 15T，总使用约 12T，可用约 2.1T，使用率 86%。项目目录当前约 12M，`datasets/`、`archives/` 和 `logs/` 仅有预留目录，没有实际数据、索引或 checkpoint 文件。

### 主机内存

```text
Mem: 251Gi total, 41Gi used, 58Gi free, 151Gi buff/cache, 189Gi available
Swap: 0B
```

该节点具备约 252GiB 主机内存，适合承担数据解码、OCR/layout 预处理、manifest 生成和本地检索索引构建。系统没有 Swap；后续数据处理仍应按实际 RSS 观察，不把 Swap 当作容量保障。

### GPU / CUDA

`nvidia-smi`：

```text
NVIDIA-SMI 555.58.02
Driver Version: 555.58.02
CUDA Version: 12.5
GPU 0: NVIDIA GeForce RTX 4090, 24564 MiB
GPU 1: NVIDIA GeForce RTX 4090, 24564 MiB
```

当时两张卡均约使用 15MiB、GPU 利用率 0%。Torch 检查结果：

```text
torch.__version__: 2.4.1+cu124
torch.cuda.is_available: True
torch.cuda.device_count: 2
cuda.device[0]: NVIDIA GeForce RTX 4090
cuda.device[1]: NVIDIA GeForce RTX 4090
```

### Python / ML 包

```text
Python 3.8.5
torch: installed (2.4.1+cu124)
transformers: not installed
vllm: not installed
ray: not installed
deepspeed: not installed
datasets: not installed
```

## 3. 对 E0 数据准备的最小建议

1. 当前节点可作为 E0 小规模数据中枢，先准备 20 条 perception 和 20 条 knowledge 样本，不启动全量数据下载。
2. 在记录数据源、license、版本、split key 和 `group_id` 规则后，再生成 E0 manifest；先按来源/文档/图片实体切分，再生成 OCR、bbox、crop 和反事实派生物。
3. 数据侧优先准备 root image、root 坐标系 bbox、答案和本地 Search 所需的最小 snippet/evidence 字段；第一版不引入嵌套 crop 或 PDF 翻页数据协议。
4. 该节点有足够主机内存执行后续 OCR/layout 和小型 BM25 索引实验，但磁盘使用率已达 86%，应在下载任何公开数据前登记大小、保留空间和清理路径。
5. 当前没有 Transformers/vLLM/Ray/DeepSpeed/datasets，依赖组合应等待双节点环境报告和 E0 兼容性决策后统一规划；本轮不在数据节点安装 ML 栈。

## 4. 结论与边界

数据节点硬件与职责匹配：两张 24GB 级 GPU 可用于离线视觉/OCR 或小规模处理，约 252GiB RAM 适合数据预处理和本地索引；它不是主训练节点。当前基线没有发现需要立即处理的 GPU、内存或 Python 包异常，但磁盘使用率较高且无 Swap，应将数据规模和归档路径纳入后续 manifest/下载计划审计。
