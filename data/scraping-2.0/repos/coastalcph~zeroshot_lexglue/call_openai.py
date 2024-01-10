import json
import os.path
import random

import openai
import tqdm
import argparse
from data import DATA_DIR
from build_instructions.templates import TEMPLATES
random.seed(42)

def main(args):
    # Provide OpenAI API key
    api_key = input("Please provide an OpenAI API key:\n")
    openai.api_key = api_key
    OPTIONS_PRESENTATION_TEXT = TEMPLATES[args.dataset_name]['OPTIONS_PRESENTATION_TEXT']
    QUESTION_TEXT = TEMPLATES[args.dataset_name]['QUESTION_TEXT']
    dataset = []
    label_wise_dataset = {}
    with open(os.path.join(DATA_DIR, f'{args.dataset_name}.jsonl')) as in_file:
        for line in in_file:
            sample_data = json.loads(line)
            dataset.append(sample_data)
            if args.few_shot_k:
                for label in sample_data['answer'].split(','):
                    if label in label_wise_dataset:
                        label_wise_dataset[label.lower().strip()].append(' '.join(sample_data['input_text'].split(OPTIONS_PRESENTATION_TEXT)[0].split(' ')[:args.truncate_demonstrations])
                                                                         + QUESTION_TEXT + ' ' + sample_data['answer'])
                    else:
                        label_wise_dataset[label.lower().strip()] = [' '.join(sample_data['input_text'].split(OPTIONS_PRESENTATION_TEXT)[0].split(' ')[:args.truncate_demonstrations])
                                                                     + QUESTION_TEXT + ' ' + sample_data['answer']]

    predictions = []
    if not args.few_shot_k and os.path.exists(os.path.join(DATA_DIR, f'{args.dataset_name}_{args.model_name}_predictions.jsonl')):
        with open(os.path.join(DATA_DIR, f'{args.dataset_name}_{args.model_name}_predictions.jsonl')) as in_file:
            for line in in_file:
                predictions.append(json.loads(line))

    demonstration_text = ''
    if args.few_shot_k:
        random_labels = random.sample(list(label_wise_dataset.keys()), k=args.few_shot_k)
        demos = [random.sample(label_wise_dataset[label], k=1)[0] for label in random_labels]
        demonstration_text = '\n\n'.join(demos) + '\n\n'

    for idx, example in tqdm.tqdm(enumerate(dataset)):
        if len(predictions) and predictions[idx]['prediction'] is not None:
            dataset[idx]['prediction'] = predictions[idx]['prediction']
            print(f'Predictions for example #{idx} is already available!')
            continue
        if args.model_name == 'gpt-3.5-turbo':
            try:
                response = openai.ChatCompletion.create(
                  model=args.model_name,
                  messages=[
                        {"role": "user", "content": demonstration_text + example['input_text']},
                    ],
                  max_tokens=100
                )
                dataset[idx]['prediction'] = response['choices'][0]['message']['content']
            except Exception as inst:
                print(inst)
                dataset[idx]['prediction'] = None
        else:
            try:
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=demonstration_text + example['input_text'],
                    max_tokens=100
                )
                dataset[idx]['prediction'] = response['choices'][0]['message']['content']
            except:
                dataset[idx]['prediction'] = None

    name_extension = f'_few_shot-{args.few_shot_k}' if args.few_shot_k else ''
    folder_name = f'_few_shot-predictions' if args.few_shot_k else 'zero-shot-predictions'
    with open(os.path.join(DATA_DIR, folder_name, f'{args.dataset_name}_{args.model_name}_predictions{name_extension}.jsonl'), 'w') as file:
        for example in dataset:
            file.write(json.dumps(example) + '\n')


parser = argparse.ArgumentParser(description='Prompting GPT')
parser.add_argument("--dataset_name", type=str, default='ledgar', help="Name of dataset as stored on HF")
parser.add_argument("--model_name", type=str, default='gpt-3.5-turbo', help="GPT model name")
parser.add_argument("--few_shot_k", type=int, default=8, help="Number of k-shots")
parser.add_argument("--truncate_demonstrations", type=int, default=100, help="Truncation of demonstrations")


args = parser.parse_args()

main(args)
