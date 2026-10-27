import json
import requests
import pandas as pd
import csv
import os

def classify(model, prompt):
    response = requests.post("http://localhost:11434/api/generate", json={'model': model,
                                                                            'prompt': prompt,
                                                                            'stream': False,
                                                                            'options': {'temperature': 0,
                                                                                        'seed': 42,
                                                                                        'num_predict': 10
                                                                                        }}) 
    result = response.json()['response']
    return result   

template = '''Determine whether this email is spam. SPAM includes unwanted, unsolicited advertising and fraudulent emails. The email contains no punctuation and is in lowercase.
Answer in one word only: SPAM or NOT_SPAM.
Email: {text}
Answer:'''

def parser(answer):
    answer = answer.lower().strip()
    if answer.startswith('not'):
        result = 0
        return result
    elif answer.startswith('spam'):
        result = 1   
        return result
    else:
        return None 

def save_row(row, filename):
# Функция заводит csv файл (если его нет) и записывает в него метрики модели после одного запроса    
    file_exists = os.path.exists(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()      
        writer.writerow(row)      

data = pd.read_csv('data/phishing_test_set.csv')
def run_quality(model):    
    for i, row in data.iterrows():
        prompt = template.format(text=row['text_combined'])
        answer = classify(model, prompt)
        row_out = {
            'model': model,
            'email_id': i,
            'true_label': row['label'],
            'raw_answer': answer,
            'predicted': parser(answer),
        }
        save_row(row_out, 'llm_quality.csv')

run_quality('llama3.2:1b')        
run_quality('llama3.2:1b-instruct-q4_K_M')
run_quality('llama3.2:3b')
run_quality('qwen2.5:1.5b')     