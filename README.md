# Ocean Acoustic Surrogate

针对一个刻意收窄、真实数据锚定的验收域，学习从深海声速剖面和距离相关地形到二维
传播损失场的快速代理：

- GEBCO 2026 巴士海峡候选区四条平滑低维地形，水深包络 2000--4800 m；
- WOA23 6 月温盐经 TEOS-10 转换，并施加小幅平滑 SSP 变化；
- 沙质流体半空间：1700 m/s、2000 kg/m³、0.8 dB/λ；
- 声源深度 50 m、频率 1000 Hz、距离 50 km；
- 25,600-ray Bellhop 非相干 TL 标签，输出网格 96×256；
- batch=1 热态完整推理 P95 不超过 100 ms；
- 密封同分布测试集 TL RMSE 不超过 2 dB。

项目优先复用 `ocean-acoustic-agent` 的数值后端。大型 Bellhop case、数据集、
checkpoint 和完整日志保存在环境变量 `OCEAN_SURROGATE_ROOT` 指向的目录，Git 仅保存
代码、配置、轻量结果、图和报告。

## 已完成结果

256 条 25,600-ray Bellhop 标签已全部成功生成并冻结，按 192/32/32 划分训练、验证和
密封测试集。全局训练均值场测试 RMSE 为 5.625 dB；最终 Global FNO-L 将其降至
0.703 dB（下降 87.49%），MAE 为 0.261 dB，独立 A100 完整推理 P95 为 4.47 ms。
1.33 M 参数的 Global FNO-S 达到 RMSE 0.843 dB、CPU P95 50.11 ms。

面向甲方的任务定义、数据来源与构造、改进型 FNO 方法、实验协议和结果分析见
[`docs/Ocean_Acoustic_Surrogate_Technical_Report_v1.4.pdf`](docs/Ocean_Acoustic_Surrogate_Technical_Report_v1.4.pdf)；
可编辑 LaTeX 源码和图表位于 [`docs/technical_report/`](docs/technical_report/)。详细工程记录另见
[`docs/project_report.md`](docs/project_report.md)。

复用 `ocean-field-project` 的 GEBCO 2026 地形、WOA23 月度 SSP 和现成 Bellhop 标签所做的
真实环境快速验证见
[`docs/field_validation_bashi_reuse_v0.1.md`](docs/field_validation_bashi_reuse_v0.1.md)。该结果
不新增 Bellhop 计算，并明确区分 500 Hz/3,200 射线 proxy 与正式 1 kHz 验收。

## 快速开始

面向学生和甲方机器的一键入口默认只检查环境与测试，不运行 Bellhop：

```bash
REPRO_MODE=check bash scripts/reproduce_mvp.sh

# 已放置冻结数据后，只训练并复核最终模型
REPRO_MODE=train bash scripts/reproduce_mvp.sh

# 仅在确认数值环境后，显式从零生成 Bellhop 标签并训练
REPRO_MODE=full bash scripts/reproduce_mvp.sh
```

等价的分步命令如下：

```bash
uv sync --locked
uv run pytest

# 小型收敛审计
uv run ocean-acoustic-surrogate pilot configs/realistic_terrain_mvp.yaml --samples 8

# 生成 256 场正式数据
uv run ocean-acoustic-surrogate generate configs/realistic_terrain_mvp.yaml --samples 256

# 运行冻结实验组
uv run ocean-acoustic-surrogate campaign \
  configs/realistic_terrain_mvp.yaml configs/realistic_campaign.yaml --samples 256

# 独立重载 checkpoint，在指定设备复核冻结测试集
uv run ocean-acoustic-surrogate verify \
  configs/realistic_terrain_mvp.yaml /path/to/run --samples 256 --device cuda
```
