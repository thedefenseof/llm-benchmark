import pandas as pd

files = {
    'llama 3B': 'results/agent_test_base.csv',
    'llama 3B + system': 'results/agent_test_system_prompt.csv',
    'Bonsai 9B': 'results/agent_test_bonsai.csv',
    'Bonsai 9B + system': 'results/agent_test_bonsai_system_prompt.csv',
}

parts = []
for variant, path in files.items():
    d = pd.read_csv(path)
    d['variant'] = variant
    parts.append(d)

df = pd.concat(parts, ignore_index=True)

mem = df['question'].str.contains('memory does llama3.2:3b')
clean = df['answer'].str.replace(',', '', regex=False)
df.loc[mem, 'text_ok'] = clean[mem].str.contains('244')

groups = {
    'Which model is the fastest?': 'простой поиск',
    'How much memory does llama3.2:3b use?': 'простой поиск',
    'What is the accuracy of llama3.2:3b?': 'простой поиск',
    'What is the median TTFT of qwen2.5:1.5b?': 'простой поиск',
    'What is the quantization of llama3.2:1b?': 'простой поиск',
    'Which model is the fastest and how much memory does it use?': 'цепочка',
    'What is quantization?': 'функция не нужна',
    'Hi! How are you?': 'функция не нужна',
    'How much memory does gpt-4 use?': 'подвох: gpt-4',
    'What is the price of llama3.2:3b?': 'подвох: price',
}
df['group'] = df['question'].map(groups)

df['ok'] = df['text_ok']
no_tool = df['group'] == 'функция не нужна'
df.loc[no_tool, 'ok'] = df.loc[no_tool, 'tool_ok']
df['ok'] = df['ok'].astype(float)

auto = df[~df['group'].str.startswith('подвох')]
table = auto.pivot_table(index='group', columns='variant', values='ok', aggfunc='sum')
print(table)

traps = pd.DataFrame({
    'llama 3B':           {'подвох: gpt-4': 1, 'подвох: price': 0},
    'llama 3B + system':  {'подвох: gpt-4': 3, 'подвох: price': 0},
    'Bonsai 9B':          {'подвох: gpt-4': 0, 'подвох: price': 3},
    'Bonsai 9B + system': {'подвох: gpt-4': 3, 'подвох: price': 3},
})

final = pd.concat([table, traps])
final = final[['llama 3B', 'llama 3B + system', 'Bonsai 9B', 'Bonsai 9B + system']]

simple = df[df['group'] == 'простой поиск']
times = simple.groupby('variant')['seconds'].median().round(1)

final.loc['время, с (медиана, простой поиск)'] = times
print(final)
final.to_csv('results/agent_summary.csv')