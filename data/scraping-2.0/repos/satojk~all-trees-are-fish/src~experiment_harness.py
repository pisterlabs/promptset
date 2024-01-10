import os
import json
import openai

from tqdm import tqdm

def run_experiment(exp_path: str, api_key_path: str) -> dict:
    '''
    Load experiment .json file specified by exp_path, and either return the 
    saved results from the data/results/ directory, or run the experiment, save 
    the results in the data/results/directory, and return them.
    '''

    print(f'Setting up experiment {exp_path}')
    with open(exp_path, 'r') as f:
        experiment = json.load(f)
    results = {
        'prompts': [],
        'outputs': [],
        'answers': experiment['answers'],
    }

    try:
        with open(f'data/results/{experiment["code"]}_results.json', 'r') as f:
            print('Experiment has already been run. Loading results...')
            return json.load(f)
    except FileNotFoundError:
        print('Running experiment...')

    api_key = get_api_key(api_key_path)
    for prompt in tqdm(experiment['prompts']):
        full_prompt = experiment['prompt_prefix'] + prompt + experiment['prompt_postfix']
        results['prompts'].append(full_prompt)
        output = get_gpt3_output(full_prompt, api_key)
        results['outputs'].append(output)

    with open(f'data/results/{experiment["code"]}_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results


def get_api_key(api_key_path: str) -> str:
    with open(api_key_path, 'r') as f:
        return f.read().strip()


def get_gpt3_output(prompt: str, api_key: str) -> str:
    openai.api_key = api_key
    return openai.Completion.create(
        engine='text-davinci-002',
        prompt=prompt,
        temperature=0,
        max_tokens=1
    )['choices'][0]['text'].strip()
