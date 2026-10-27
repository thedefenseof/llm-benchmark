import pandas as pd

df = pd.read_csv('results/agent_test_bonsai_system_prompt.csv')

print(df.groupby('question').agg(
    tool_ok=('tool_ok', 'mean'),
    text_ok=('text_ok', 'mean'),
    seconds=('seconds', 'median')
).round(2))

for _, row in df[df['question'].str.contains('gpt-4|price')].iterrows():
    print(row['run'], row['calls'])
    print(row['answer'])
    print('---')