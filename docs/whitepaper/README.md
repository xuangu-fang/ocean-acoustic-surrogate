# 白皮书构建说明

白皮书正文使用 XeLaTeX 排版，全部图表由冻结的数据、预测结果和指标 JSON 自动生成。

```bash
cd /home/ubuntu/project/ocean-acoustic-surrogate
OCEAN_SURROGATE_ROOT=/home/ubuntu/ocean-acoustic-surrogate-artifacts \
  bash scripts/build_whitepaper.sh
```

最终 PDF 输出到：

`docs/Ocean_Acoustic_Surrogate_Whitepaper_v1.0.pdf`

正文源文件为 `whitepaper.tex`，图表源程序为
`scripts/generate_whitepaper_figures.py`。图表的 PDF/PNG 双格式版本位于 `assets/`。
