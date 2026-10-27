import pandas as pd
import requests
import csv
import os
import json

def ask(model, prompt, num_predict):
    # Функция для отправки промптов моделям
    response = requests.post("http://localhost:11434/api/generate", json={'model': model,
                                                                            'prompt': prompt,
                                                                            'stream': False,
                                                                            'options': {'temperature': 0,
                                                                                        'seed': 42,
                                                                                        'num_predict': num_predict
                                                                                        }}) 

    result = response.json()['response']
    return result 

def is_correct(answer, expected):
    # Функция сравнивает ожидаемый ответ и реальный ответ модели
    answer = answer.lower().strip(".!?, \n")
    expected = expected.lower()
    return(int(answer == expected))
    
def cyrillic_percentage(answer):
    # Функция считает % кириллицы в ответах. Полезно для использования в длинных русских ответах, тк в них модели часто переходят на другой язык
    cyrillic_count = 0
    total_letters = 0

    for char in answer:
        if char.isalpha():
            total_letters += 1
            if 'А' <= char.upper() <= 'Я' or char.lower() in 'ё':
                cyrillic_count += 1
    if total_letters == 0:
        return None
    return (cyrillic_count / total_letters) * 100 

def save_row(row, filename):
# Функция заводит csv файл (если его нет) и записывает в него метрики модели после одного запроса    
    file_exists = os.path.exists(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()      
        writer.writerow(row)           

NUM_PREDICT = {'fact_tf': 10, 'math_tf': 10, 'geo': 20, 'long': 200}    
tasks = pd.read_csv("data/ru_en_tasks.csv", sep=';', encoding='cp1251')

def run_short(model):
    # Функция запускает прогон моделей по промптам с ожидаемым коротким ответом и отдает csv c подсчитанными результатами 
    for i, row in tasks.iterrows():
        if row['type'] == 'long':
            continue

        answer = ask(model, row['prompt'], NUM_PREDICT[row['type']])

        row_out = {
            'model': model,
            'id': row['id'],
            'type': row['type'],
            'lang': row['lang'],
            'raw_answer': answer,
            'expected': row['expected'],
            'correct': is_correct(answer, row['expected']),       
            'cyrillic_pct': cyrillic_percentage(answer),   
        }
        save_row(row_out, 'ru_en_results.csv')          

def ask_long(model, prompt):
     # Функция запускает прогон моделей по промптам с ожидаемым коротким ответом и отдает csv c подсчитанными результатами
    response = requests.post("http://localhost:11434/api/generate",
                             json={'model': model,
                                   'prompt': prompt,
                                   'stream': True,
                                   'options': {'temperature': 0,
                                               'seed': 42,
                                               'num_predict': NUM_PREDICT['long']}},
                             stream=True)
    text = ''
    output_tokens = None
    tok_per_s = None
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        text += chunk.get('response', '')
        if chunk.get('done'):
            output_tokens = chunk['eval_count']
            tok_per_s = output_tokens / (chunk['eval_duration'] / 1e9)
    return {'text': text, 'output_tokens': output_tokens, 'tok_per_s': tok_per_s}

def run_long(model, repeats=3):
     # Функция запускает прогон моделей по промптам с ожидаемым длинным ответом по 3 раза и отдает csv c подсчитанными результатами
    for i, row in tasks.iterrows():
        if row['type'] != 'long':
            continue
        for run in range(1, repeats + 1):
            res = ask_long(model, row['prompt'])
            row_out = {
                'model': model,
                'id': row['id'],
                'lang': row['lang'],
                'run': run,
                'tok_per_s': round(res['tok_per_s'], 2),
                'output_tokens': res['output_tokens'],
                'chars': len(res['text']),
                'cyrillic_pct': cyrillic_percentage(res['text']),
                'text': res['text'],
            }
            save_row(row_out, 'ru_en_long.csv')

for m in ['llama3.2:1b', 'llama3.2:1b-instruct-q4_K_M', 'llama3.2:3b', 'qwen2.5:1.5b']:
    run_long(m)            