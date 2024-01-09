import os, json

import pandas as pd

import openai

import math

import datetime


opposite = { 'no' : 'yes',
            'yes' : 'no' }


def test_GPT3(prompt, answer):
    prompt = prompt
    
    response = openai.Completion.create(
        model=config['model'],
        prompt=prompt,
        temperature=0,
        max_tokens=2,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=5
    )
    
    top_logits = response['choices'][0]['logprobs']['top_logprobs'][0].to_dict()
    top_percents = { key[1:] : math.exp(top_logits[key]) for key in top_logits }
    top_predictions = sorted(top_percents, key=top_percents.get, reverse=True)
    
    return top_predictions


config_file = 'exp_config_sports_understanding.json'

config = None

with open(os.path.join(os.getcwd(), config_file), 'r') as f_config:
    config = json.load(f_config)

if not config:
    print(f"Couldn't load config file: {config_file}")
    exit()


openai.api_key = config['openai_key']

df_data = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'testing', 'sports_understanding', config['eval_file']), index_col=0)

test_cases = []

for i, row in enumerate(df_data.iterrows()):
    if not i % 10:
        print('.', end='')
    
    prompt = row[1]['prompt']
    answer = row[1]['completion'].replace('.','').strip()
    
    top_predictions = (test_GPT3(prompt, answer))
    
    try:
        idx_answer = top_predictions.index(answer)
    except:
        idx_answer = 6
    
    try:
        idx_wrong = top_predictions.index(opposite[answer])
    except:
        idx_wrong = 6
    
    if idx_answer < idx_wrong:
        test_cases.append({'prompt' : prompt, 'answer' : answer,\
                           'top_predictions' : top_predictions, 'score' : True})
    else:
        test_cases.append({'prompt' : prompt, 'answer' : answer,\
                           'top_predictions' : top_predictions, 'score' : False})


df_experiment = pd.DataFrame(test_cases)

exp_namespace = f"{config['exp_name']} at {datetime.datetime.now()}"
os.mkdir(exp_namespace)
df_experiment.to_csv(os.path.join(os.getcwd(), exp_namespace, 'results.csv'))


with open(os.path.join(os.getcwd(), exp_namespace, 'score.txt'), 'w') as f_score:
    num_hits = len(df_experiment[df_experiment['score']])
    num_test_cases = len(df_experiment)
    score_text = f"Score on { num_test_cases } test cases: { num_hits / num_test_cases }"
    print(f"\n{score_text}\n")
    f_score.write(score_text)

