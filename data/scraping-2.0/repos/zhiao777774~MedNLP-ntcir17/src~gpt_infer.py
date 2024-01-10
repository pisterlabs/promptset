import os
import time
import argparse
import json
import openai
import ast
import pandas as pd
from pprint import pprint
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv

load_dotenv()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatcompletion_with_backoff(**kwargs) -> openai.ChatCompletion:
    return openai.ChatCompletion.create(**kwargs)


def gpt_infer(input_text: str, model='gpt-3.5-turbo') -> str:
    openai.api_key = os.getenv('OPENAI_API_KEY')

    messages = [{'role': 'user', 'content': input_text}]
    completion = chatcompletion_with_backoff(model=model,
                                             max_tokens=300,
                                             temperature=0,
                                             messages=messages)

    res = completion.choices[0].message.content
    return res


def recognize_synptoms_with_drug_information(data, prompt, model):
    with open('src/data/sideeffects/medlineplus_side_effects.json', 'r') as f:
        tmp_side_effect = json.load(f)
    side_effect = {k: v for item in tmp_side_effect for k, v in item.items()}

    results = []
    for i, item in data.iterrows():
        text = item['text']
        drugs = ast.literal_eval(item['drug_name'])
        drug_side_effects = {
            d: side_effect[d]['normal']
            for d in drugs if d != 'None'
        }

        if not drug_side_effects:
            drug_side_effects = 'None'

        input_text = prompt.replace('{input}',
                                    text).replace('{drugs}', drug_side_effects)

        print(i, '=' * 30)
        pprint(input_text)
        print('->')

        res = gpt_infer(input_text, model)
        results.append(str(res))

        pprint(res)
        print()
        time.sleep(0.2)

    return results


def recognize_synptoms(data, prompt, model):
    results = []

    for i, item in data.iterrows():
        text = item['text']
        input_text = prompt.replace('{input}', text)

        print(i, '=' * 30)
        pprint(input_text)
        print('->')

        res = gpt_infer(input_text, model)
        results.append(str(res))

        pprint(res)
        print()
        time.sleep(0.2)

    return results


def recognize_adverse_reactions(data, prompt, model):
    results = []

    for i, item in data.iterrows():
        text = item['text']
        input_text = prompt.replace('{input}', text)

        print(i, '=' * 30)
        pprint(input_text)
        print('->')

        res = gpt_infer(input_text, model)
        results.append(str(res))

        pprint(res)
        print()
        time.sleep(0.2)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',
                        type=str,
                        default='src/data/preproc/preproc_en_train.csv')
    parser.add_argument('--prompt_file',
                        type=str,
                        default='src/prompts/recognize_symptoms.prompt')
    parser.add_argument('--output_folder', type=str, default='src/output/')
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4',
                        choices=['gpt-3.5-turbo', 'gpt-4'])
    parser.add_argument('--task',
                        type=str,
                        default='rs',
                        choices=['rs', 'rswd', 'rar'])
    args = parser.parse_args()

    data = pd.read_csv(args.data_file)
    with open(args.prompt_file, 'r') as f:
        prompt = f.read()

    task = args.task
    if task == 'rs':
        task_func = recognize_synptoms
    elif task == 'rswd':
        task_func = recognize_synptoms_with_drug_information
    else:
        task_func = recognize_adverse_reactions

    results = task_func(data.copy(), prompt, args.model)

    data['gpt_response'] = results
    data.to_json(
        f'{args.output_folder}/preproc_en_train_{task}_{args.model}.json',
        orient='records')
