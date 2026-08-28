# 技术报告构建说明

技术报告正文使用 XeLaTeX 排版，全部图表由冻结的数据、预测结果和指标 JSON 自动生成。

```bash
cd /home/ubuntu/project/ocean-acoustic-surrogate
bash scripts/build_technical_report.sh
```

若数据和实验产物位于其他位置，可通过 `OCEAN_SURROGATE_ROOT=/path/to/artifacts`
显式覆盖默认的大数据根目录。

默认构建 256 场真实数据锚定的 V1.4 技术报告，最终 PDF 输出到：

`docs/Ocean_Acoustic_Surrogate_Technical_Report_v1.4.pdf`

附录中的一键复现入口为仓库根目录下的 `scripts/reproduce_mvp.sh`。默认不执行 Bellhop；
只有显式设置 `REPRO_MODE=full` 才会生成标签并重新训练。

V1.4 正文源文件为 `technical_report_v1.4.tex`，图表源程序为
`scripts/generate_realistic_report_figures.py`。图表的 PDF/PNG 双格式版本位于 `assets/`。
V1.2/V1.3 源文件继续保留，用于追溯此前报告。
