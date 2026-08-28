# Ocean Acoustic Surrogate

面向固定典型深海场景的二维声传播损失快速代理模型。项目以高射线数 Bellhop 非相干
传播损失（Transmission Loss, TL）为参考标签，使用改进型二维 Fourier Neural Operator
（FNO）学习

```math
\{\text{SSP }c(z),\ \text{海底地形 }b(r)\}
\longmapsto \text{TL 场 }Y(z,r),
```

在当前冻结任务域内，最终模型将全局训练均值场的测试 RMSE 从 **5.625 dB** 降至
**0.703 dB**，独立 A100 batch=1 完整推理 P95 为 **4.47 ms**，同时通过
2 dB 精度门槛和 100 ms 时延门槛。

> 当前目标是先在参数严格、标签高质量、范围受控的任务域内形成可复现闭环；本项目不宣称
> 已具备跨频率、跨海区、任意底质或未见地形类别的通用外推能力。

## 冻结任务

| 项目 | 设置 |
|---|---|
| 频率与声源 | 1000 Hz；声源深度 50 m |
| 传播区域 | 0.1--50 km，共 256 个距离点 |
| 接收深度 | 10--1990 m，共 96 个深度点 |
| 地形 | GEBCO 2026 巴士海峡候选区；4 条经筛选的平滑低维剖面 |
| 水深包络 | 2000--4800 m |
| SSP | WOA23 6 月温盐，经 TEOS-10 转换并施加小幅平滑变化 |
| 海底介质 | 流体沙质半空间：1700 m/s、2000 kg/m³、0.8 dB/λ |
| 参考标签 | Bellhop 非相干 TL；25,600 rays/field |
| 输出 | `96 × 256` TL 场，单位 dB |
| 验收条件 | 测试 RMSE/MAE ≤ 2 dB；batch=1 P95 ≤ 100 ms |

## 冻结结果

正式数据集包含 256 个完整环境场，每条地形 64 场，分层划分为
train/validation/test = 192/32/32。前 128 个已冻结高射线标签经逐项核验后复用，另外
生成 128 个新 Bellhop 标签；生成成功率为 256/256。

| 方法 | 参数量 | 测试 RMSE | 测试 MAE | P90 单样本 RMSE | 推理 P95 |
|---|---:|---:|---:|---:|---:|
| 全局训练均值场 | -- | 5.625 dB | 3.857 dB | 6.386 dB | -- |
| Global FNO-S | 1.33 M | 0.843 dB | 0.383 dB | 1.122 dB | GPU 3.93 ms；CPU 50.11 ms |
| **Global FNO-L** | **6.30 M** | **0.703 dB** | **0.261 dB** | **0.922 dB** | **A100 4.47 ms** |

Global FNO-L 相对全局均值场的 RMSE 降幅为 **87.49%**；最差测试样本 RMSE 为
1.129 dB，高梯度区域 RMSE 为 1.936 dB。独立进程重新加载 checkpoint 后复算 RMSE
为 0.703443 dB，与训练产物记录值相差小于 `2.4e-5 dB`。

Global FNO-L 的 CPU 完整推理诊断 P95 为 133.72 ms，因此最终大模型以 A100 GPU 为
时延验收平台；只有 CPU 的部署环境可选用上表中的紧凑 Global FNO-S。

冻结数据集 SHA-256：

```text
b66f4f367b93ef6f84802604e1d7f25e85d1b6a20a224252e6776b1285ef1d37
```

轻量冻结结果见
[`docs/results/realistic_v0.6_n256_verification_summary.json`](docs/results/realistic_v0.6_n256_verification_summary.json)。

## 方法概览

模型只使用 SSP、距离相关海底地形以及内部坐标通道，不使用人工 Hankel 特征。单样本的
主要张量流为：

```text
SSP [41] + terrain [256]
  -> interpolate / normalize / replicate
physical input [2, 96, 256]
  -> append normalized depth and range coordinates
coordinate-aware input [4, 96, 256]
  -> pointwise lift 4 -> 32
hidden field [32, 96, 256]
  -> replicate padding (+8 depth, +16 range)
padded field [32, 104, 272]
  -> 4 × {anisotropic spectral path + local path + skip connection}
  -> crop -> pointwise projection 32 -> 128 -> 1
normalized residual [1, 96, 256]
  -> global train-only mean field + residual scale
predicted TL [96, 256] dB
```

相对基础 FNO，当前模型的关键设计包括：

1. 将距离相关地形作为显式输入通道，使网络能够感知海底边界；
2. 使用深度/距离各向异性模态数 `Kz=16, Kr=48`，为更长的距离轴保留更多频率；
3. 在非周期声场边界上使用复制 padding，减轻 FFT 绕回伪影；
4. 每个谱模块并联 `1×1` 局部路径，并加入 GroupNorm 和层间残差；
5. 预测训练集全局均值场上的标准化残差，再解码回 dB 域。

完整符号、公式、模块输入输出和实验分析见
[`docs/Ocean_Acoustic_Surrogate_Technical_Report_v1.4.pdf`](docs/Ocean_Acoustic_Surrogate_Technical_Report_v1.4.pdf)。

## 仓库结构

```text
ocean-acoustic-surrogate/
├── configs/
│   ├── realistic_terrain_mvp.yaml   # 冻结场景、SSP、地形、Bellhop 与划分
│   └── realistic_campaign.yaml      # Global FNO-S/L 与数据量消融
├── src/ocean_acoustic_surrogate/
│   ├── ssp.py                       # 嵌套 Latin hypercube 与 SSP 构造
│   ├── dataset.py                   # Bellhop 任务、收敛审计、断点生成与打包
│   ├── features.py                  # SSP/地形特征和训练域目标变换
│   ├── models/fno.py                # 各向异性二维 FNO
│   ├── training.py                  # 训练、验证选模、测试和时延
│   └── verification.py              # 独立 checkpoint 重载与密封测试复核
├── scripts/
│   ├── reproduce_mvp.sh             # 一键检查、训练、全流程和复核
│   └── build_technical_report.sh    # 重建图表与 V1.4 PDF
├── docs/results/                     # 可提交 Git 的冻结指标摘要
├── docs/technical_report/            # LaTeX 源文件和图表
└── tests/                            # 配置、特征、模型、数据与指标测试
```

项目复用兄弟仓库 `ocean-acoustic-agent` 的 Bellhop 数值后端。大型 case、数据集、checkpoint
和完整日志不写入 Git，而是统一保存到 `OCEAN_SURROGATE_ROOT`。

## 环境准备

要求 Python 3.11、[`uv`](https://docs.astral.sh/uv/) 以及同级目录中的
`ocean-acoustic-agent`：

```text
acoustic-work/
├── ocean-acoustic-agent/
└── ocean-acoustic-surrogate/
```

默认大数据目录为：

```bash
export OCEAN_SURROGATE_ROOT=/mnt/data/xuangu-fang/ocean-acoustics/projects/ocean-acoustic-surrogate
```

如果 `ocean-acoustic-agent` 不在默认兄弟目录，可显式设置：

```bash
export OCEAN_AGENT_ROOT=/absolute/path/to/ocean-acoustic-agent
```

## 一键复现

### 1. 只检查代码和环境

默认模式不会运行 Bellhop，也不会训练模型：

```bash
REPRO_MODE=check bash scripts/reproduce_mvp.sh
```

该命令同步锁定依赖，并运行 Ruff 和完整测试集。

### 2. 使用已有冻结数据训练最终模型

先将冻结 `dataset.npz` 放到配置所解析的 `n256/` 目录，然后运行：

```bash
REPRO_MODE=train bash scripts/reproduce_mvp.sh
```

默认训练 `real_r2_terrain_fno`（Global FNO-L），随后在独立流程中复核密封测试集。

### 3. 只复核已有 checkpoint

```bash
export REPRO_RUN_DIR="$OCEAN_SURROGATE_ROOT/runs/<run_id>"
REPRO_MODE=verify bash scripts/reproduce_mvp.sh
```

### 4. 从 Bellhop 标签开始执行完整流程

以下模式会检查 Bellhop、运行射线数收敛、生成或续跑 256 个标签、训练并复核，成本明显
高于前三种模式：

```bash
REPRO_MODE=full bash scripts/reproduce_mvp.sh
```

扩容时如已有与新设计前缀完全一致的数据集，可安全复用并逐项校验：

```bash
export REPRO_REUSE_PREFIX_FROM=/absolute/path/to/earlier/n128
REPRO_MODE=full bash scripts/reproduce_mvp.sh
```

可用 `REPRO_CONFIG`、`REPRO_CAMPAIGN`、`REPRO_EXPERIMENT`、`REPRO_SAMPLES` 和
`REPRO_RUN_DIR` 覆盖默认设置。运行 `REPRO_MODE=help bash scripts/reproduce_mvp.sh`
可查看全部参数。

## 分步命令

```bash
uv python install 3.11
uv sync --locked
uv run ruff check .
uv run pytest -q

# 8 场射线数收敛审计
uv run ocean-acoustic-surrogate pilot \
  configs/realistic_terrain_mvp.yaml --samples 8

# 生成或断点续跑 256 场标签
uv run ocean-acoustic-surrogate generate \
  configs/realistic_terrain_mvp.yaml --samples 256

# 输出数据统计和全局均值基线
uv run ocean-acoustic-surrogate profile \
  configs/realistic_terrain_mvp.yaml --samples 256

# 训练冻结的最终实验
uv run ocean-acoustic-surrogate campaign \
  configs/realistic_terrain_mvp.yaml configs/realistic_campaign.yaml \
  --samples 256 --only real_r2_terrain_fno

# 独立重载 checkpoint 并复核
uv run ocean-acoustic-surrogate verify \
  configs/realistic_terrain_mvp.yaml /absolute/path/to/run \
  --samples 256 --device cuda
```

## 外部产物布局

默认生成以下结构：

```text
$OCEAN_SURROGATE_ROOT/
├── datasets/
│   └── bashi_gebco_selected_diverse_woa23_june_v0.6_n256/
│       └── n256/
│           ├── dataset.npz
│           ├── manifest.json
│           └── ...
├── runs/
│   └── <timestamp>_<experiment>/
│       ├── model.pt
│       ├── metrics.json
│       ├── history.json
│       ├── predictions.npz
│       └── independent_verification_<device>.json
└── campaigns/
    └── latest.json
```

一次可验收复现至少应保留 `dataset.npz`、`manifest.json`、`model.pt`、`metrics.json`、
`history.json`、`predictions.npz` 和独立验证 JSON。

## 报告与参考材料

- [甲方技术报告 V1.4（PDF）](docs/Ocean_Acoustic_Surrogate_Technical_Report_v1.4.pdf)
- [V1.4 LaTeX 源码与构建说明](docs/technical_report/)
- [256 场冻结验证摘要](docs/results/realistic_v0.6_n256_verification_summary.json)
- [完整工程实验记录](docs/project_report.md)
- [早期复用现有标签的巴士海峡快速验证](docs/field_validation_bashi_reuse_v0.1.md)

重建报告：

```bash
bash scripts/build_technical_report.sh
```

当前发布版本为 Git tag `technical-report-v1.4`。
