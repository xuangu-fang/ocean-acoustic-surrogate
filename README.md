# Ocean Acoustic Surrogate

针对一个刻意收窄的验收域，学习从深海声速剖面到二维传播损失场的快速代理：

- 水深 2000 m、距离无关环境；
- 沙质流体半空间：1700 m/s、2000 kg/m³、0.8 dB/λ；
- 声源深度 50 m、频率 1000 Hz、距离 50 km；
- Bellhop 非相干 TL 标签；
- batch=1 热态完整推理 P95 不超过 100 ms；
- 密封同分布测试集 TL RMSE 不超过 2 dB。

项目优先复用 `ocean-acoustic-agent` 的数值后端。大型 Bellhop case、数据集、
checkpoint 和完整日志保存在环境变量 `OCEAN_SURROGATE_ROOT` 指向的目录，Git 仅保存
代码、配置、轻量结果、图和报告。

## 已完成结果

512 条 25,600-ray Bellhop 标签已全部成功生成并冻结。五轮实验的精度优胜模型在密封
测试集达到 RMSE 0.5165 dB、MAE 0.1802 dB，A100 完整推理 P95 4.73 ms；1.33 M
参数的小模型在独立 CPU 复核中达到 RMSE 0.7582 dB、P95 38.32 ms。两者均显著通过
2 dB / 100 ms 门槛。完整方法、负结果和适用边界见
[`docs/project_report.md`](docs/project_report.md)。

## 快速开始

```bash
uv sync --locked
uv run pytest

# 小型收敛审计
uv run ocean-acoustic-surrogate pilot configs/mvp.yaml --samples 8

# 生成 512 场正式数据
uv run ocean-acoustic-surrogate generate configs/mvp.yaml --samples 512

# 运行全部模型迭代
uv run ocean-acoustic-surrogate campaign configs/mvp.yaml configs/campaign.yaml

# 独立重载 checkpoint，在指定设备复核冻结测试集
uv run ocean-acoustic-surrogate verify configs/mvp.yaml /path/to/run --device cuda
```
