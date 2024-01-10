import json
import jsonlines
import openai
import rouge
import time
import pprint


# Constants
DEBUG = False
max_tokens = 250
temperature = 0.0
training_filename = 'briankeng-2023-06-09-finetune.jsonl'
model_list_filename = 'finetune.json'

# Load data
with open(model_list_filename) as f:
    data = json.load(f)

results = []
for entry in data['data']:
    succeeded = (entry['status'] == 'succeeded')
    if not succeeded:
        continue

    bjlkengbot = entry['fine_tuned_model'].startswith('curie:ft-personal:bjlkengbot')
    if not bjlkengbot:
        continue

    result = {}
    result['model'] = entry
    print('-' * 80)
    print(f'id: {entry["id"]}')
    print(f'model: {entry["model"]}')
    print(f'fine_tuned_model: {entry["fine_tuned_model"]}')
    print(f'status: {entry["status"]}')
    print(f'hyperparams: {entry["hyperparams"]}')

    # Load jsonl file
    training_data = []
    with jsonlines.open(training_filename) as reader:
        for obj in reader:
            training_data.append(obj)


    data = []
    for i, obj in enumerate(training_data):
        # Fine tuning structure:
        # - All prompts end with suffix `\n\n###\n\n`
        # - All prompts start with prefix `QUESTION: `
        # - All completions end with suffix ` END`
        completion = openai.Completion.create(model=entry['fine_tuned_model'], 
                                              prompt=obj['prompt'], 
                                              max_tokens=max_tokens, 
                                              temperature=temperature, 
                                              stop=[' END'])

        # Remove " END" from source
        datum = {}
        reference = obj['completion'][:-4]
        hypothesis = completion['choices'][0]['text']
        datum['reference'] = reference
        datum['hypothesis'] = hypothesis
        datum['training_data'] = obj


        rscore = rouge.Rouge().get_scores(hypothesis, reference)[0]
        datum['rouge'] = rscore
        data.append(datum)

        if DEBUG:
            pprint.pprint(datum)
            break

        # Sleep every 10 requests
        if (i + 1) % 10 == 0:
            print(f'Sleeping after {i} calls...')
            time.sleep(15)

    result['data'] = data    
    results.append(result)
    if DEBUG:
        pprint.pprint(result)
        break

# Write out JSON
with open('measure-results.json', 'w') as f:
    json.dump(results, f, indent=4)