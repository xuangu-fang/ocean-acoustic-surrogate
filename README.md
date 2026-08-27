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
```

权威方法、实验和验收结果见 `docs/project_report.md`；该文档在实验完成后由冻结结果更新。

