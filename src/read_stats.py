import pandas as pd
stats = pd.read_csv("results/llm_stats.csv")
print(stats.groupby('model').size())
print(stats.groupby('model')[['tok_per_s', 'ttft', 'mem_mb']].median())