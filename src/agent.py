import pandas as pd
import time

def fastest_model():
    df = pd.read_csv('results/summary.csv')
    best = df.loc[df['tok_per_s'].idxmax()]
    return f'Fastest model: {best["model"]}, speed {best["tok_per_s"]:.1f} tok/s'

def get_model_stat(model, metric):
    df = pd.read_csv('results/summary.csv')
    row = df[df['model'] == model]
    if row.empty:
        return 'Модель не найдена'
    if metric not in df.columns:
        return f'Метрика не найдена. Доступно: {list(df.columns)}'
    return row.iloc[0][metric]


import requests

URL = 'http://localhost:11434/api/chat'

tools = [{
    'type': 'function',
    'function': {
        'name': 'fastest_model',
        'description': "The function determines which model is the fastest in terms of tokens per second.",
        'parameters': {'type': 'object', 'properties': {}}
    }
},
{
    'type': 'function',
    'function': {
        'name': 'get_model_stat',
        'description': 'The function receives the model name and the metric of interest and returns the value corresponding to that metric for the requested model.',
        'parameters': {
            'type': 'object',
            'properties': {
                'model': {'type': 'string', 'description': 'The name of the large language model'},
                'metric': {'type': 'string', 'description': 'Metric name. Acceptable names: ttft_median, ttft_p95, itl_median, tok_per_s, mem_mb, accuracy, precision, recall, quantization'}
            },
            'required': ['model', 'metric']
        }
    }
}]

SYSTEM_PROMPT = "Use functions only for questions about the results of model benchmarks. Answer the other questions (greetings, general concepts) yourself. If the function returned an error or said that there is no data, say so. Don’t make up numbers. For a question about the fastest model and its characteristics, first call fastest_model, then get_model_stat with the found name."
available = {'fastest_model': fastest_model, 'get_model_stat': get_model_stat}

def ask_agent(question, model):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question}
        ]
    calls_made = []
    for step in range(5):
        response = requests.post(URL, json={'model': model, 'messages': messages, 'tools': tools, 'stream': False, 'options': {'temperature': 0}})
        answer = response.json()['message']
        messages.append(answer)

        if not answer.get('tool_calls'):
            return answer['content'], calls_made

        for call in answer['tool_calls']:
            name = call['function']['name']
            args = call['function']['arguments']
            calls_made.append(name)
            result = available[name](**args)
            messages.append({'role': 'tool', 'tool_name': name, 'content': str(result)})

    return 'Не уложился в 5 шагов', calls_made

models = ['bonsai-2:9b']

questions = [
    ('Which model is the fastest?', ['fastest_model'], '1b-instruct-q4_K_M'),
    ('How much memory does llama3.2:3b use?', ['get_model_stat'], '244'),
    ('What is the accuracy of llama3.2:3b?', ['get_model_stat'], '0.8'),
    ('What is the median TTFT of qwen2.5:1.5b?', ['get_model_stat'], '0.8'),
    ('What is the quantization of llama3.2:1b?', ['get_model_stat'], 'Q8_0'),
    ('Which model is the fastest and how much memory does it use?', ['fastest_model', 'get_model_stat'], '963'),
    ('What is quantization?', [], None),
    ('Hi! How are you?', [], None),
    ('How much memory does gpt-4 use?', ['get_model_stat'], None),
    ('What is the price of llama3.2:3b?', ['get_model_stat'], None)
]

total = len(models) * len(questions) * 3
rows = []
for model in models:
    for q, expected_tools, expected_text in questions:
        for run in range(3):
            start = time.time()
            text, calls = ask_agent(q, model)
            seconds = time.time() - start
            rows.append({
                'model': model,
                'question': q,
                'run': run,
                'calls': ', '.join(calls),
                'text_ok': (expected_text in text) if expected_text else None,
                'tool_ok': all(t in calls for t in expected_tools) if expected_tools else len(calls) == 0,
                'seconds': round(seconds, 1),
                'answer': text
            })
            done = len(rows)
            print(f'{done}/{total} готово: {q[:40]} (запуск {run}), {seconds:.1f} с')

pd.DataFrame(rows).to_csv('results/agent_test_bonsai_system_prompt.csv', index=False)