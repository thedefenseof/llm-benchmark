import json
import requests
import time
import numpy
import csv
import os
import pandas as pd

def run_once(model, prompt): 
# Функция запускает модель один раз, замеряет основные метрики и возвращает словарь с ними
    t_start = time.perf_counter()

    response = requests.post("http://localhost:11434/api/generate", json={'model': model,
                                                                        'prompt': prompt,
                                                                        'stream': True,
                                                                        'options': {'temperature': 0,
                                                                                    'seed': 42,
                                                                                    'num_predict': 200
                                                                                    }}, stream=True)

    t_first = None
    times = []

    for line in response.iter_lines():
        chunk = json.loads(line)

        if t_first is None and chunk["response"]:
            t_first = time.perf_counter()
        if chunk['response']:
            times.append(time.perf_counter()) 
        if chunk["done"]:
            eval_count = chunk['eval_count']
            eval_duration = chunk['eval_duration']
            prompt_eval_count = chunk['prompt_eval_count']

    diffs = [times[i] - times[i - 1] for i in range(1, len(times))]  
    
    ttft = t_first - t_start
    itl_mean = sum(diffs) / len(diffs) * 1000
    itl_p95 = float(numpy.percentile(diffs, 95) * 1000)
    tok_per_s = eval_count / (eval_duration / 1e9)

    result = {
        'ttft': ttft,
        'itl_mean': itl_mean,
        'itl_p95': itl_p95,
        'tok_per_s': tok_per_s,
        'prompt_tokens': prompt_eval_count,
        'output_tokens': eval_count
    }

    return result

def get_memory_mb(model):
# Функция замеряет квантизацию и занимаемую память, возвращает словарь с этими значениями 
    data = requests.get('http://localhost:11434/api/ps').json()

    for i in data['models']:
        if i['name'] == model:
            return {
                'mem_mb': round(i['size'] / (1024 * 1024), 2),
                'quantization': i['details']['quantization_level']
            }
    return None 

def save_row(row, filename):
# Функция заводит csv файл (если его нет) и записывает в него метрики модели после одного запроса    
    file_exists = os.path.exists(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()      
        writer.writerow(row)   

# Набор промптов разной длины для тестирования моделей
prompts = {
    'short_1': 'Tell me what is DDL in SQL',
    'short_2': 'Explain difference between DWH and DataLake',
    'short_3': 'Describe who is data engineer',
    'medium_1': '''Tell me what skills are needed to work in a team: 
                We are the Machine Learning Technology Research team. We are engaged in data analysis and machine learning for threat detection tasks: phishing, spam, malware, suspicious activity in corporate networks, and so on. 
                We also implement machine learning technologies (including those based on LLM) in our products and carry out projects to optimize the company’s business processes. 
                Among our projects is the Kaspersky LLM Service, an internal service based on large language models for the secure processing of confidential data.
                We are looking for those who want to learn:
                1. Support large language model inference services
                2. Research benchmarking and performance optimization capabilities of LLMs
                3. Build informative dashboards and analyze service performance
                4. Select optimal LLMs based on the tasks being performed and available resources
                5. Study Azure DevOps pipelines, OpenShift, Kubernetes, Docker''',
    'medium_2': '''Analyze the composition of the chips and provide consumption recommendations: 
                Potatoes, vegetable oil, flavoring (salt, flavoring agents (onion powder, parsley leaves, garlic powder, cheese powder, sour cream powder), whey powder, sucrose, flavor enhancer (monosodium glutamate 1-substituted)).
                The product may contain a small amount of fish and/or seafood powder.
                Ready to use
                Nutritional value per 100 g:
                • Proteins – 6 g
                • Fats – 30 g
                • Carbohydrates – 53 g
                Energy value per 100 g: 510 kcal
                Energy value of 1 pack: 255 kcal''',
    'medium_3': '''Solve the riddle and explain the solution: 
                You enter an old, abandoned, haunted house. It has neither electricity nor running water. Once inside, you see three doors in front of you. Each door has a number on it. 
                Behind each door, there is one way to die. Behind door number one, you will die because there is a lion inside and it will eat you. Behind door number two, you will die because there is a zombie inside and it will attack you. 
                Behind door number three, you will die in the electric chair.
                You have no way to turn back, so you must go through some door. Which door will you go through?''',
    'long_1': '''Analyze the songs texts: I created juice, it’s not clear, as if I’m in “8 Mile” (erdo)
                You’re writing comments, I wish you were dead
                Killed with a knife, bro, just buried
                I don’t give a fuck about all your rap if your rap is going to Amiri
                I’m wearing clothes from the year you were born
                The fact that you’re a whore is public opinion
                We’re up high, they say it’s just luck
                After one meeting with you, I’m not the same anymore
                (Fuck, fuck, ps-s) They’re pissing their pants because of us
                Fuckboy, we’re smoking on you, and then we’ll piss on you from above
                It’s hard for me to carry two heads
                I’ve seen a lot of porn where you starred, uh‑woah‑woah
                Take the money and transfer it to my card
                The fact that you’re with us, bro, is a fucking expensive rider
                I can’t sleep at night, I’m writing my first personal “Barter”
                My love is dead, but it wakes up once a quarter
                My slime can no longer rock a mix of indie and slick
                You can stop bragging about the fact that you jerked me off
                We’re still the same, Barvikha and beer
                So, what’s up, how are you? I saved up from lunches at ERD, damn, ha‑ha (hey)
                Marlboro red in my pocket, I’m a cowboy
                I feel like a villain, but she says, “Hero.”
                My bro is fucked up right now, if you speak Russian, he’s alive.
                Baby, leave me a knife, I want to die a star, hey‑woah.
                I created some juice, I don’t understand, it’s like I’m in *8 Mile*.
                You’re writing comments, I wish you were killed.
                Killed with a knife, bro, just buried.
                I don’t care about all your rap if your rap goes to Amiri.
                I created a juice, I don’t understand, it’s like I’m in “8 Mile”
                You’re writing comments, I wish you were dead
                Killed with a knife, bro, just buried you
                I don’t give a fuck about all your rap, I don’t care, I don’t care
                I’m kin’ing you the same way, just a little different.
                I was too shy to film everything, okay, just cover your face with your hand.
                I look in the mirror the same way, but I won’t be myself.
                I love Moscow, but I’m sick of all the Moscow sluts.
                I see beauty in ugliness, my beauty is in ugliness.
                Now we’re on “you,” because you’ve long been someone else.
                Let me cut you, and you shoot me in the forehead.
                We’re psychopaths (whoa), flexing like we’re going to die tomorrow.
                But that’s unlikely, I caught a phew, and the white smoke is somehow darkening the ceiling.
                I quit for a while, now…
                I’m smoking like I never knew how to, feeling like I’ve gone fucking stupid.
                You’re not shooting, let me give you the sight, the sum is big, like I’m a retard.
                Bro, I can’t think today, you’re still there, but not mine.
                You’re still here, but I’m not there. We’re doing a lot, but honestly, I know I’ll be misunderstood.
                Dynasty!
                First million, I’m the first champion, and my skin is tight from the mouth.
                Vilchinsky knows what he’s doing, we’ll film it on an iPhone.
                Concerts from Ufa, we’re recording this on Samson.
                I remember my mom said, “Throw the microphone away”
                I’m the first champion, and my skin’s tightened by my mouth.
                The creator knows what he’s doing, we’ll shoot it on an iPhone (yeah).
                Concerts from Moscow, we’re writing this on Samson.
                I remember my mom said, “Throw the microphone away”
                I could have stayed a shadow, and everything would have been the same.
                But I wanted to be louder than these high‑rise buildings.
                My shit turned into mumble, so break it down into quotes.
                Is your friend also a rapper? But he’s selling Zolo-, and (whoa)
                Sorry, but I’m a bit different right now.
                If I forgive you now, I won’t forgive myself later.
                Yes, my diagnosis is abnormal, but a psychologist can lie.
                I remember showing you my rap, but you didn’t care.
                And what about now? It’s funny how you’re asking for a photo with me.
                No meet and greet, I can максимум give a throat‑slit (m, a)
                It’s become normal for me to drop half a million on a T‑shirt
                My favorite rapper in the past was stealing my ideas (oops)
                First million, I’m the first champion, and my mouth pulled the skin tight
                Vilchinsky knows what’s up, we’ll film it on an iPhone
                Concerts from Ufa, we’re recording this on Samson
                I remember my mom used to say: “Throw the microphone away”
                I’m the first champion, and my skin was pulled tight by my mouth.
                The creator knows what’s up, we’ll film it on an iPhone.
                Concerts from Moscow, we’re recording this on Samson.
                I remember my mom used to say: “Throw the microphone away”''',
    'long_2': '''Summarize and put it in simple language:
                Now that you have a clearer understanding of the data sharing and cross-domain
                collaboration models, I’d like to share some key principles that you should keep in
                mind to ensure your distributed data management strategy is successful. I’ll start with
                some high-level advice here, then get deeper into specifics for data domains and data
                providers:
                Avoid data silos.
                The first design principle is to discourage the use of silos or integration hubs
                for data distribution. Data silos require additional transformation steps, which
                makes the chain between provider and consumers longer and much harder to
                manage. In, for example, the data warehouse model, we have to deal with 
                inconsistencies when integrating all the data because bounded contexts often conflict.
                Data providers and data consumers in the data warehouse model are tightly
                coupled, which makes the dependencies much more difficult to manage. 
                Organizations often can’t trust data silos because of data quality issues, corrections, and
                inconsistencies.
                By eliminating data silos for data distribution and using newer and more modern
                forms of data integration and distribution, we avoid teams bypassing the integration obstacles 
                and driving the architecture into an unmanageable point-to-pointarchitecture.
                Connecting your domains using point-to-point interfaces creates exponential 
                complexity and should be avoided at all costs.
                For example, interconnecting 100 applications results in 5,000
                point-to-point connections. For 1,000 applications, there will
                be roughly 500,000 point-to-point connections. Point-to-point
                connections can’t provide the control and agility enterprises
                need because the sheer number of communication channels
                makes it nearly impossible to oversee all dependencies.
                Only capture and modify data at the golden source.
                The second principle is aimed at increasing trust in data. It’s important to ensure
                you’re capturing only unique data from your golden sources before distributing
                any data to other systems. Changes to original data should be made only at the
                golden source within the domain, with approval from the owner. 
                For transparency and trust, it’s important to make your golden sources publicly known, so I
                encourage you to set up a central register or catalog that keeps track of all unique
                golden source systems.
                50 | Chapter 2: Organizing Data Using Data Domains Respect the rules of data ownership.
                When applications consume and use data, they can become new golden sources.
                In this circumstance the data truly has a new context, so there are new facts. This
                brings us to our next principle: when data has been changed and new facts are
                created, new ownership is expected. Simple “syntactic transformations,” such as
                filters, unions, upper- to lowercase conversions, and field renames do not require
                new ownership, since all the facts remain the same. This is strongly linked to
                another foundational principle: don’t distribute any data you don’t own. When
                distributing data that originates in and is owned by another domain, always
                ensure your domain gets approval from that domain.''',
    'long_3': '''Compare this language with SQL and tell me which one is better:
                Data Analysis Expressions (DAX) is the native formula and query language for Microsoft PowerPivot, Power BI Desktop and SQL Server Analysis Services (SSAS) Tabular models. 
                DAX includes some of the functions that are used in Excel formulas with additional functions that are designed to work with relational data and perform dynamic aggregation. 
                It is, in part, an evolution of the Multidimensional Expression (MDX) language developed by Microsoft for Analysis Services multidimensional models (often called cubes) combined with Excel formula functions. 
                It is designed to be simple and easy to learn, while exposing the power and flexibility of PowerPivot and SSAS tabular models.
                The Data Analysis Expressions (DAX) language provides a specialized syntax for querying Analysis Services tabular model. 
                DAX is not a programming language. DAX is primarily a formula language and is also a query language. 
                You can use DAX to define custom calculations for Calculated Columns, Measures, Calculated Tables, Calculation Groups, Custom Format Strings, and filter expressions in role-based security in Tabular models. 
                The same Analysis Services engine for Tabular models is also used in Power BI and Power Pivot for Excel. Power BI also uses DAX for conditional formatting expressions and other dynamic properties of visual components.
                DAX can compute values for seven data types:

                Integer (called Whole Number in Power BI, a signed 64-bit integer)
                Decimal (called Decimal Number in Power BI, a double-precision (64-bit) floating-point number)
                Currency (called Fixed Decimal Number in Power BI, a number with four decimal places and stored internally as an integer divided by 10,000)
                DateTime (called Date/time, Date or Time in Power BI, stored internally as a floating-point number where the integer part represents the date in number of days since 30 December 1899 and the decimal part represents the time in terms of its fraction of a day)
                Boolean (called True/false in Power BI, either the value TRUE or the value FALSE)
                String (called Text in Power BI, a sequence of Unicode (UTF-16) characters, comparisons between strings in DAX are case-insensitive)
                Variant (the type given to a DAX expression that might return a value of different types depending on the conditions)
                The BLOB (binary large object) data type is managed by the Tabular model but cannot be directly manipulated by DAX expressions.
                DAX has a powerful type-handling system so that you do not have to worry much about data types. 
                When you write a DAX expression, the resulting type is based on the type of the terms used in the expression and on the operator used. 
                Type conversion happens automatically during the expression evaluation. 
                However, if an expression might return different data types depending on the conditions, then the data type of the expression is defined as Variant.'''
}

def run_benchmark(model):
    # Функция прогревает модель на тестовом промпте и потом собирает данные о ней, прогоняя по 5 раз по промптам    
    
    # Прогрев модели
    run_once(model, "Say hello")

    # Записываем память после прогрева
    memory = get_memory_mb(model)

    # Гоняем модель 5 раз...
    for name, text in prompts.items():
        for i in range(1, 6): 
            unique_text = f'[{int(time.time())}] {text}'                      
            result = run_once(model, unique_text)
            row = {
                'prompt_id': name,
                'run': i,
                'model': model,
                'quantization': memory['quantization'],
                'prompt_tokens': result['prompt_tokens'],
                'output_tokens': result['output_tokens'],
                'ttft': round(result['ttft'], 3),
                'itl_mean': round(result['itl_mean'], 3),
                'itl_p95': round(result['itl_p95'], 3),
                'tok_per_s': round(result['tok_per_s'], 3),
                'mem_mb': memory['mem_mb']
            }
            save_row(row, 'llm_stats.csv')
            print(name, i, 'Готово')

run_benchmark('llama3.2:1b')        
run_benchmark('llama3.2:1b-instruct-q4_K_M')
run_benchmark('llama3.2:3b')
run_benchmark('qwen2.5:1.5b') 
