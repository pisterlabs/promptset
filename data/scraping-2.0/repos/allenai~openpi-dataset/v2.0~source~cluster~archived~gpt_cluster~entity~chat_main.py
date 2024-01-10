import os
import ast
import json
import openai
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from openai_inference import chat_inference
from utils import load_json, sample_data, save_json, load_txt, clean_steps

np.random.seed(42)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_path', type=str, required=True, help='path to save results'
    )
    parser.add_argument(
        '--api_path', type=str, required=True, help='path to the api key'
    )
    return parser.parse_args()


def load_template(template_path: str) -> str:
    return ''.join(open(template_path, 'r').readlines())


def parse_result(res: dict) -> list:
    content = res['content']
    print(content)
    content = content.split('\n')[1:]
    content = [item.replace('- ', '') for item in content]
    return content


def main():
    args = get_args()
    data = load_json('../../../../data/dev-ranked.json')
    header_template = load_template('../assets/templates/entity_chat_header_v2.txt')
    content_template = load_template('../assets/templates/entity_chat_template_v2.txt')

    examples = load_json('../assets/examples/entity_chat_example.txt')

    openai.api_key_path = args.api_path

    # build examples
    example_prompt = ast.literal_eval(header_template)
    for entry in examples.values():
        cur_goal = entry['goal']
        cur_steps = '\n'.join(entry['steps'])
        cur_entities = ', '.join(entry['entities']).replace('"', "'")
        cur_clusters = '- ' + '\\n- '.join(entry['clusters']).replace('"', "'").replace(')', '').replace('(', '')
        example_prompt += ast.literal_eval(content_template.replace('{entities}', cur_entities).replace('{grouped_entities}', cur_clusters))
    
    results = {}
    for key, entry in tqdm(data.items()):
        cur_goal = entry['goal']
        cur_steps = '\n'.join(entry['steps'])
        cur_states = entry['states']

        original_cur_entities = [item['entity'] for item in cur_states]
        cur_entities = [item.split(' | ') for item in original_cur_entities]
        cur_entities = ', '.join([list for sublst in cur_entities for list in sublst])

        cur_prompt = ast.literal_eval(content_template.replace('{goal}', cur_goal).replace('{entities}', cur_entities))[0]

        cur_input = example_prompt + [cur_prompt]

        out = chat_inference(cur_input, 'gpt-3.5-turbo')

        results[key] = {
            'original_entities': original_cur_entities,
            'input_entities': cur_entities,
            'grouped_entities': parse_result(out),
        }

    with open(args.out_path, 'w') as f:
        json.dump(results, f, indent=4)
    f.close()

if __name__ == '__main__':
    main()
