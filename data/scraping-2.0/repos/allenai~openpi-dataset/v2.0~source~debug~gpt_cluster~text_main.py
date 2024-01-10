import os
import ast
import json
import openai
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from openai_inference import gpt3_inference, chat_inference
from utils import load_json, sample_data, save_json, load_txt, clean_steps

np.random.seed(42)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to the un-clustered file'
    )
    parser.add_argument(
        '--out_path', type=str, required=True, help='path to save results'
    )
    parser.add_argument(
        '--template_path', type=str, required=True, help='path to the prompt'
    )
    parser.add_argument(
        '--examples_path', type=str, required=True, help='path to in-context examples'
    )
    parser.add_argument(
        '--api_path', type=str, required=True, help='path to the api key'
    )
    return parser.parse_args()


def load_template(template_path: str) -> str:
    return ''.join(open(template_path, 'r').readlines())


def main():
    args = get_args()
    data = load_json(args.data_path)
    template = load_txt(args.template_path)

    examples = load_json(args.examples_path)

    openai.api_key_path = args.api_path

    # build examples
    example_prompt = ''
    for entry in examples.values():
        cur_goal = entry['goal']
        cur_steps = '\n'.join(entry['steps'])
        cur_entities = ', '.join(entry['entities'])
        cur_clusters = ', '.join(entry['clusters'])
        example_prompt += template.replace('{goal}', cur_goal)  \
                                  .replace('{steps}', cur_steps)  \
                                  .replace('{entity_states}', cur_entities)  \
                                  .replace('{clustered_entity_states}', cur_clusters)
        example_prompt += '\n\n\n'
    
    results = []
    for idx, entry in enumerate(tqdm(data.values())):
        if idx >= 5:
            break
        cur_goal = entry['goal']
        cur_steps = '\n'.join(entry['steps'])
        cur_states = entry['states']

        original_cur_entities = [item['entity'] for item in cur_states]
        cur_entities = [item.split(' | ') for item in original_cur_entities]
        cur_entities = ', '.join([list for sublst in cur_entities for list in sublst])

        cur_prompt = template.replace('{goal}', cur_goal)  \
                             .replace('{steps}', cur_steps)  \
                             .replace('{entity_states}', cur_entities)  \
                             .replace('{clustered_entity_states}', '')

        cur_input = example_prompt + cur_prompt
        out = gpt3_inference(cur_input, 'text-davinci-003')
        results.append({
            'original_entities': original_cur_entities,
            'input_entities': cur_entities,
            'grouped_entities': out,
        })
    
    with open(args.out_path, 'w') as f:
        json.dump(results, f, indent=4)
    f.close()

if __name__ == '__main__':
    main()
