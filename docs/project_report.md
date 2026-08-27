# 1 kHz 深海声传播损失快速代理项目报告

状态：实验执行中  
日期：2026-08-27

## 1. 目标与验收域

本项目不以跨海区、跨设备或跨任务复用为目标。第一阶段只证明：在一个固定且可复现的
典型深海环境族中，代理模型可以把声速剖面快速映射到完整距离—深度传播损失场，并满足：

- 频率 1000 Hz；
- 最大距离 50 km；
- Bellhop 非相干 TL 作为数值参考；
- 测试集整体 RMSE 与 MAE 均不超过 2 dB；
- batch=1 热态完整推理 P95 不超过 100 ms。

固定参数为水深 2000 m、声源深度 50 m、距离无关环境，以及声速 1700 m/s、密度
2000 kg/m³、衰减 0.8 dB/λ 的沙质流体半空间。第一阶段仅改变 SSP。

## 2. 学生实验参考

学生补充实验使用 3060 条南海东侧 6 月仿真 SSP、64 通道、72 个 Fourier modes、
4 个 Fourier 层、Hankel 幅度附加特征、batch size 40 和 2000 个 epoch；随机 9:1
划分后的测试 RMSE 约 1.1–1.2 dB。其误差主要分布在会聚区和多射线路径叠加位置。

本项目把这组结果视为可达性证据和架构参考，不把 3060 样本、2000 epochs 或随机拆分
视为必须复制的设置。实验从更小的数据量开始，并以冻结测试指标决定是否扩展。

## 3. 数据生成方法

权威配置为 `configs/mvp.yaml`。SSP 使用一个围绕给定 6 月深海剖面的四维平滑族：全局
声速偏移、上层温跃层幅度、声道轴垂向移动和深层梯度。使用 Latin hypercube 在冻结
边界内采样，不加入像素级随机噪声、不同源深、不同底质或距离相关结构。

标签通过 `ocean-acoustic-agent` 的 Bellhop 后端生成。每个样本保存 SSP、参数、TL、有效
掩膜、求解耗时、完整 Bellhop case 路径、配置哈希和依赖 Git SHA。大文件位于：

`/mnt/data/xuangu-fang/ocean-acoustics/projects/ocean-acoustic-surrogate/`

### 3.1 标签收敛审计

在 8 个独立 SSP 上分别运行 3200、6400、12800 和 25600 rays。相邻层级的平均
RMSE 分别为 0.104、0.066 和 0.052 dB；12800 与 25600 rays 的最差逐场 RMSE 为
0.141 dB，8 个样本无失败。该误差比最终 2 dB 门槛低一个数量级以上，因此正式标签
固定为 25600 rays。热态单场中位生成时间约 9.46 s。

本结果也说明，学生使用的非相干 TL 远比此前项目中的半相干场对射线数量稳定；因此
本项目不混用两类 TL，也不把半相干实验的收敛结论外推到当前标签。

### 3.2 正式数据集

实验结果待写入。

## 4. 模型与损失

基础模型为各向异性 FNO2d：深度和距离方向使用不同的截断模态数。输入为沿距离复制的
SSP，部分迭代增加归一化 Hankel 传播幅度。网络内部加入物理坐标。输出不直接拟合整个
TL 动态范围，而是预测相对训练集逐网格平均 TL 的残差；平均场只使用训练集计算。

基础损失是在有效 Bellhop 网格上的归一化 MSE。后续迭代依次考察：

1. 增加距离方向 Fourier modes；
2. 非周期 padding 与残差块；
3. Hankel 特征与局部空间卷积分支；
4. 梯度损失与高梯度会聚区加权。

所有模型最终都使用相同、未加权的原始 dB 指标评估。

## 5. 实验迭代与 Loss

实验结果待写入。

## 6. 最佳模型验收

实验结果待写入。

## 7. 结论、失败模式与适用边界

实验结果待写入。

## 8. 复现命令

```bash
uv sync --locked --group dev
uv run pytest
uv run ocean-acoustic-surrogate pilot configs/mvp.yaml --samples 8
uv run ocean-acoustic-surrogate generate configs/mvp.yaml --samples 512
uv run ocean-acoustic-surrogate campaign configs/mvp.yaml configs/campaign.yaml --samples 512
```
