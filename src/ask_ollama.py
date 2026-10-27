import requests
import os

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

response = requests.post(
    f'{OLLAMA_URL}/api/generate',
    json={
        'model': 'llama3.2:1b',
        'prompt': 'What is database?',
        'stream': False,
    },
)

result = response.json()

print('Токенов:', result['eval_count'])
print('Секунд на генерацию:', result['eval_duration'] / 1e9)
print('Скорость генерации:', round(result['eval_count'] / (result['eval_duration'] / 1e9), 2))

os.makedirs('results', exist_ok=True)

with open('results/docker_test.txt', 'a', encoding='utf-8') as f:
    f.write(f"tokens={result['eval_count']}, tok_per_s={round(result['eval_count'] / (result['eval_duration'] / 1e9), 2)}\n")