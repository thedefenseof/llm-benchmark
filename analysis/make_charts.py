import matplotlib.pyplot as plt
import pandas as pd

stats = pd.read_csv('results/llm_stats.csv')

speed = stats.groupby('model').agg(
    ttft_median=('ttft', 'median'),
    ttft_p95=('ttft', lambda s: s.quantile(0.95)),
    itl_median=('itl_mean', 'median'),
    tok_per_s=('tok_per_s', 'median'),
    mem_mb=('mem_mb', 'median'),
    quantisation=('quantization', 'first')
)

quality = pd.read_csv('results/quality_summary.csv')

summary = pd.merge(speed.reset_index(), quality[['model', 'accuracy', 'precision', 'recall']], on='model')

plt.figure(figsize=(7, 5))
plt.scatter(summary['tok_per_s'], summary['accuracy'])

offsets = {
    'llama3.2:3b': (8, 8),
    'llama3.2:1b': (0, -20),
    'qwen2.5:1.5b': (0, 12),
    'llama3.2:1b-instruct-q4_K_M': (0, -20),
}

for _, row in summary.iterrows():
    plt.annotate(row['model'], (row['tok_per_s'], row['accuracy']),
                 xytext=offsets[row['model']], textcoords='offset points',
                 ha='center')

plt.xlabel('Токенов в секунду')
plt.ylabel('Accuracy на классификации спама')
plt.title('Скорость и качество')
plt.xlim(10, 50)
plt.ylim(0.4, 0.9)
plt.axhline(0.5, linestyle='--', color='gray')
plt.text(11, 0.51, 'уровень угадывания', color='gray')
plt.grid(alpha=0.3)
plt.savefig('results/charts/quality_vs_speed.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 4))
s = summary.sort_values('tok_per_s')
plt.barh(s['model'], s['tok_per_s'])
plt.xlabel('Токенов в секунду (медиана)')
plt.title('Скорость генерации по моделям')
plt.grid(axis='x', alpha=0.3)
plt.savefig('results/charts/speed_by_model.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 4))
s = summary.sort_values('mem_mb')
s['label'] = s['model'] + ' (' + s['quantisation'] + ')'
plt.barh(s['label'], s['mem_mb'])
plt.xlabel('Память (МБ)')
plt.title('Память по моделям')
plt.grid(axis='x', alpha=0.3)
plt.savefig('results/charts/memory_by_model.png', dpi=150, bbox_inches='tight')
plt.show()

t = stats.groupby(['model', 'prompt_id'])[['prompt_tokens', 'ttft']].median().reset_index()

plt.figure(figsize=(8, 5))
for m, g in t.groupby('model'):
    plt.plot(g['prompt_tokens'], g['ttft'], marker='o', label=m)

plt.xlabel('Длина промпта (токенов)')
plt.ylabel('TTFT, с (медиана)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('results/charts/ttft_vs_prompt.png', dpi=150, bbox_inches='tight')
plt.show()