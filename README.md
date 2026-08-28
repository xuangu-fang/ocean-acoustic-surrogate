# Ocean Acoustic Surrogate

面向固定典型深海场景的二维声传播损失快速代理模型。项目以高射线数 Bellhop 非相干
传播损失（Transmission Loss, TL）为参考标签，使用 **SeaBAR-FNO**（Seasonal and
Bathymetry-Aware Residual Fourier Neural Operator）学习

```math
\{\text{SSP }c(z),\ \text{海底地形 }b(r)\}
\longmapsto \text{TL 场 }Y(z,r).
```

在当前冻结任务域内，SeaBAR-FNO 将全局训练均值场测试 RMSE 从 **4.954 dB** 降至
**0.702 dB**，相对下降 **85.83%**；独立 A100、batch=1 完整推理 P95 为
**4.70 ms**。精度和时延均通过 2 dB / 100 ms 门槛。

> 当前目标是在参数严格、标签高质量、范围受控的任务域内形成可复现闭环；不宣称已具备
> 跨频率、跨海区、任意底质或未见地形类别的通用外推能力。

## 冻结任务与数据

| 项目 | 设置 |
|---|---|
| 频率与声源 | 1000 Hz；声源深度 50 m |
| 传播区域 | 0.1--50 km，共 256 个距离点 |
| 接收深度 | 10--1990 m，共 96 个深度点 |
| 地形 | GEBCO 2026 巴士海峡候选区；4 条经筛选的平滑低维剖面 |
| SSP | WOA23 3/6/12 月温盐，经 TEOS-10 转换并施加受控平滑扰动 |
| 海底介质 | 流体沙质半空间：1700 m/s、2000 kg/m³、0.8 dB/λ |
| 参考标签 | Bellhop 非相干 TL；25,600 rays/field |
| 输出 | `96 × 256` TL 场，单位 dB |
| 验收条件 | 测试 RMSE/MAE ≤ 2 dB；batch=1 P95 ≤ 100 ms |

正式数据集为 `4 地形 × 3 月份 × 32 扰动 = 384` 个完整环境场，联合分层划分为
train/validation/test = **288/48/48**；每个地形--月份组合严格包含 24/4/4 场。
生成成功率为 384/384，其中前 96 场经逐元素核验后复用，新增 288 个 Bellhop 标签。

数据集 SHA-256：

```text
9c90e051851e24082816b435bc529250f64d8a61c8a8f603916071eb0fda0a61
```

轻量冻结摘要见
[`docs/results/seasonal_v0.7_n384_verification_summary.json`](docs/results/seasonal_v0.7_n384_verification_summary.json)。

## 冻结结果

| 方法 | 参数量 | 测试 RMSE | 测试 MAE | P90 单样本 RMSE | A100 P95 |
|---|---:|---:|---:|---:|---:|
| 全局训练均值场 | -- | 4.954 dB | 3.389 dB | 6.442 dB | -- |
| 仅 SSP | 6.30 M | 4.961 dB | 3.385 dB | 6.764 dB | 4.56 ms |
| 各向同性谱核 | 2.11 M | 0.802 dB | 0.384 dB | 1.179 dB | 4.95 ms |
| 无边界保护 | 6.30 M | 0.712 dB | 0.306 dB | 1.027 dB | 4.54 ms |
| **SeaBAR-FNO** | **6.30 M** | **0.702 dB** | **0.295 dB** | **1.031 dB** | **4.70 ms** |

SeaBAR-FNO 最差测试样本 RMSE 为 1.320 dB，高梯度区域 RMSE 为 1.797 dB。按月份
分组 RMSE 为 0.658/0.740/0.706 dB（3/6/12 月），按地形分组为 0.469--0.864 dB。
独立进程重新加载 checkpoint 后复算 RMSE 为 0.702180 dB，与训练记录相差
`1.8e-6 dB`，checkpoint 一致性检查通过。

最终模型 CPU 诊断 P95 为 100.58 ms，因此正式时延验收平台为 A100。若目标机器仅有 CPU，
可评估各向同性谱核变体：本次 CPU P95 为 66.51 ms、RMSE 为 0.802 dB。

## 方法概览

SeaBAR-FNO 不使用人工 Hankel 特征。单样本张量流为：

```text
SSP [41] + terrain [256]
  -> 双环境上下文编码：插值、标准化、二维复制
physical input [2, 96, 256]
  -> 拼接深度/距离坐标，逐点升维 4 -> 32
hidden field [32, 96, 256]
  -> 非周期复制 padding (+8 depth, +16 range)
padded field [32, 104, 272]
  -> 4 × {各向异性谱路径 Kz=16, Kr=48 + 局部路径 + 残差融合}
  -> crop -> pointwise projection 32 -> 128 -> 1
normalized residual [1, 96, 256]
  -> 训练集全局均值场 + 冻结残差尺度
predicted TL [96, 256] dB
```

五个模块的定位为：双环境上下文编码、各向异性谱传播核、全局--局部残差融合、非周期
边界保护和训练域残差解码。消融显示地形编码是决定性模块；增加距离方向模态带来明确增益；
复制 padding 提供较小但方向一致的边界稳健性收益。

完整符号、公式、数据来源和实验分析见
[`docs/Ocean_Acoustic_Surrogate_Technical_Report_v1.5.pdf`](docs/Ocean_Acoustic_Surrogate_Technical_Report_v1.5.pdf)。

## 仓库结构

```text
ocean-acoustic-surrogate/
├── configs/
│   ├── realistic_seasonal_terrain_mvp.yaml  # 冻结场景、月份、地形与 Bellhop
│   └── seasonal_campaign.yaml               # SeaBAR-FNO 与模块消融
├── src/ocean_acoustic_surrogate/
│   ├── ssp.py                               # 月度模板、LHS 与平滑 SSP
│   ├── dataset.py                           # 收敛审计、断点生成与打包
│   ├── features.py                          # SSP/地形特征和训练域目标变换
│   ├── models/fno.py                        # 二维各向异性 FNO
│   ├── training.py                          # 训练、分层评分和时延
│   └── verification.py                      # 独立 checkpoint 复核
├── scripts/
│   ├── reproduce_mvp.sh                     # 检查、训练、全流程与复核
│   └── build_technical_report.sh            # 重建图表与 V1.5 PDF
├── docs/results/                             # 可提交 Git 的冻结指标摘要
├── docs/technical_report/                    # LaTeX 源文件和图表
└── tests/
```

项目复用兄弟仓库 `ocean-acoustic-agent` 的 Bellhop 后端。大型 case、数据集、checkpoint
和日志不进入 Git，统一保存到 `OCEAN_SURROGATE_ROOT`。

## 环境准备

要求 Python 3.11、[`uv`](https://docs.astral.sh/uv/) 以及同级目录中的
`ocean-acoustic-agent`：

```text
acoustic-work/
├── ocean-acoustic-agent/
└── ocean-acoustic-surrogate/
```

```bash
export OCEAN_SURROGATE_ROOT=/mnt/data/xuangu-fang/ocean-acoustics/projects/ocean-acoustic-surrogate
# 若 agent 不在默认兄弟目录：
export OCEAN_AGENT_ROOT=/absolute/path/to/ocean-acoustic-agent
```

## 一键复现

```bash
# 只同步依赖并运行 Ruff / pytest，不运行 Bellhop
REPRO_MODE=check bash scripts/reproduce_mvp.sh

# 使用已有 n384/dataset.npz 训练最终 SeaBAR-FNO 并独立复核
REPRO_MODE=train bash scripts/reproduce_mvp.sh

# 独立复核已有 checkpoint
export REPRO_RUN_DIR="$OCEAN_SURROGATE_ROOT/runs/<run_id>"
REPRO_MODE=verify bash scripts/reproduce_mvp.sh

# Bellhop 收敛、384 场生成/续跑、训练与复核全流程
REPRO_MODE=full bash scripts/reproduce_mvp.sh
```

默认配置为 `configs/realistic_seasonal_terrain_mvp.yaml`，实验为 `seabar_fno`，样本数为
384。可通过 `REPRO_CONFIG`、`REPRO_CAMPAIGN`、`REPRO_EXPERIMENT`、`REPRO_SAMPLES`、
`REPRO_RUN_DIR` 和 `REPRO_REUSE_PREFIX_FROM` 覆盖；运行
`REPRO_MODE=help bash scripts/reproduce_mvp.sh` 查看全部入口。

等价分步命令：

```bash
uv python install 3.11
uv sync --locked
uv run ruff check .
uv run pytest -q

uv run ocean-acoustic-surrogate pilot \
  configs/realistic_seasonal_terrain_mvp.yaml --samples 8
uv run ocean-acoustic-surrogate generate \
  configs/realistic_seasonal_terrain_mvp.yaml --samples 384
uv run ocean-acoustic-surrogate profile \
  configs/realistic_seasonal_terrain_mvp.yaml --samples 384
uv run ocean-acoustic-surrogate campaign \
  configs/realistic_seasonal_terrain_mvp.yaml configs/seasonal_campaign.yaml \
  --samples 384 --only seabar_fno
uv run ocean-acoustic-surrogate verify \
  configs/realistic_seasonal_terrain_mvp.yaml /absolute/path/to/run \
  --samples 384 --device cuda
```

一次可验收复现至少保留 `dataset.npz`、`manifest.json`、`convergence_report.json`、
`model.pt`、`metrics.json`、`history.json`、`predictions.npz` 和
`independent_verification_<device>.json`。

## 报告与版本

- [甲方技术报告 V1.5（PDF）](docs/Ocean_Acoustic_Surrogate_Technical_Report_v1.5.pdf)
- [V1.5 LaTeX 源码与构建说明](docs/technical_report/)
- [384 场冻结验证摘要](docs/results/seasonal_v0.7_n384_verification_summary.json)
- [完整工程实验记录](docs/project_report.md)

重建报告：

```bash
bash scripts/build_technical_report.sh
```

当前发布版本为 Git tag `technical-report-v1.5`。
