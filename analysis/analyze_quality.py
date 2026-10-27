import pandas as pd

q = pd.read_csv('results/llm_quality.csv')

def metrics(df):
    tp = ((df['true_label'] == 1) & (df['predicted'] == 1)).sum()
    fp = ((df['true_label'] == 0) & (df['predicted'] == 1)).sum()
    fn = ((df['true_label'] == 1) & (df['predicted'] == 0)).sum()
    tn = ((df['true_label'] == 0) & (df['predicted'] == 0)).sum()
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'accuracy': (tp + tn) / len(df),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else None,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else None,
        'unparsed': df['predicted'].isna().mean(),
    }

rows = []
for m, g in q.groupby('model'):
    r = metrics(g)
    r['model'] = m
    rows.append(r)

summary = pd.DataFrame(rows)
print(summary)
summary.to_csv('quality_summary.csv', index=False)    