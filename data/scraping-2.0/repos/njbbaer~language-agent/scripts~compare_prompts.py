import yaml
import random
import openai
from dotenv import load_dotenv

load_dotenv()


def complete(prompt):
    return openai.Completion.create(
        model='text-davinci-002',
        prompt=prompt,
        temperature=0,
        frequency_penalty=0.5,
        max_tokens=256,
        stop='Interviewer:',
    ).choices[0].text.strip()


def print_results(keys, results):
    if len(results) == 0:
        return
    print('\nResults:')
    for key in keys:
        score = results.count(key) / len(results)
        print(f'{key}: {score:.0%}')
    print(results)


with open('data.yml', 'r') as f:
    data = yaml.safe_load(f)

results = []
for i, variable in enumerate(data['variables']):
    print(f'# Test {i + 1}\n{variable}\n')
    keys = data['variations'].keys()
    shuffled_keys = random.sample(keys, len(keys))
    for j, key in enumerate(shuffled_keys):
        print(f'## Variation {j + 1}')
        variation = data['variations'][key]
        prompt = variation.replace('<<variable>>', variable)
        print(complete(prompt) + '\n')
    choice = input('Which variation is best? ')
    if choice != '':
        results.append(shuffled_keys[int(choice) - 1])
    print_results(keys, results)
    print()
