import json
import jsonlines
import openai
import requests
import rouge
import time
import pprint

# Constants
DEBUG = False
MAX_TOKENS = 250
TEMPERATURE = 0.0
TRAINING_FILENAME = 'briankeng-2023-06-09-finetune.jsonl'
URL = 'https://bjlkengbot.bjlkeng.io/api'


def query_api(query):
    ''' Courtesy of ChatGPT '''
    base_url = URL
    params = {'query': query}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        data = response.json()       # Parse the JSON response
        return data
    except requests.exceptions.RequestException as e:
        print('Error occurred while making the API request:', e)
        return None


result = {}
result['model'] = {'id': 'QA-prompt_only', 'model': 'QA-prompt_only'}

# Load jsonl file
training_data = []
with jsonlines.open(TRAINING_FILENAME) as reader:
    for obj in reader:
        training_data.append(obj)

data = []
for i, obj in enumerate(training_data):
    # Clean up prompt without start/end markers
    prompt = obj['prompt']
    if prompt.startswith('QUESTION: '):
        prompt = prompt[10:]
    if prompt.endswith('\n\n###\n\n'):
        prompt = prompt[:-7]
    prompt = prompt.strip()

    completion = query_api(prompt)

    # Remove " END" from source
    datum = {}
    reference = obj['completion'][:-4]
    hypothesis = completion['text']
    datum['reference'] = reference
    datum['hypothesis'] = hypothesis
    datum['training_data'] = obj

    rscore = rouge.Rouge().get_scores(hypothesis, reference)[0]
    datum['rouge'] = rscore
    data.append(datum)

    if DEBUG:
        pprint.pprint(datum)
        break

    # Sleep every request
    time.sleep(10.5)
    if (i + 1) % 10 == 0:
        print(f'* Finished {i + 1} data points')

result['data'] = data    
if DEBUG:
    pprint.pprint(result)

# Write out JSON
with open('measure-results-qa.json', 'w') as f:
    json.dump(result, f, indent=4)