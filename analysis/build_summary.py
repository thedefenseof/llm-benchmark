import pandas as pd

stats = pd.read_csv('C:/Users/mkobycheva/Desktop/аналитика/проекты/llm-benchmark/results/llm_stats.csv')

speed = stats.groupby('model').agg(
    ttft_median=('ttft', 'median'),
    ttft_p95=('ttft', lambda s: s.quantile(0.95)),
    itl_median=('itl_mean', 'median'),
    tok_per_s=('tok_per_s', 'median'),
    mem_mb=('mem_mb', 'median'),
    quantization=('quantization', 'first')
)

quality = pd.read_csv('results/quality_summary.csv')

summary = pd.merge(speed.reset_index(), quality[['model', 'accuracy', 'precision', 'recall']], on='model')
print(summary.round(2))

summary.to_csv('C:/Users/mkobycheva/Desktop/аналитика/проекты/llm-benchmark/results/summary.csv', index=False)