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


def parse_result(res: dict) -> list:
    content = res['content']
    content = content.split('\n')[1:]
    content = [item.replace('- ', '').strip() for item in content]
    return content


def main():
    args = get_args()
    data = load_json(args.data_path)
    header_template = load_template(os.path.join(args.template_path, 'attr_chat_header_v2.txt'))
    content_template = load_template(os.path.join(args.template_path, 'attr_chat_template_v2.txt'))

    examples = load_json(args.examples_path)
    entity_attr_template = lambda ent, attr: f'{attr} of {ent}'

    openai.api_key_path = args.api_path

    # build examples
    example_prompt = ast.literal_eval(header_template)
    for entry in examples.values():
        cur_goal = entry['goal']
        # cur_steps = '\n'.join(entry['steps'])
        cur_entities = ', '.join(entry['attributes']).replace('"', "'")
        cur_clusters = '- ' + '\\n- '.join(entry['clusters']).replace('"', "'").replace(')', '').replace('(', '')
        example_prompt += ast.literal_eval(content_template.replace('{attributes}', cur_entities).replace('{grouped_attr}', cur_clusters))
    
    results = {}
    for key, entry in tqdm(data.items()):
        cur_goal = entry['goal']
        cur_steps = '\n'.join(entry['steps'])
        cur_states = entry['states']

        original_cur_attr_lst = []
        for step_entry in cur_states:
            cur_entity = step_entry['entity'].split('|')[0].strip()
            for step_value in step_entry['answers'].values():
                cur_attr = [item['attribute'].split('|') for item in step_value]
                cur_attr = [lst for sublst in cur_attr for lst in sublst]
                cur_attr = [item.strip() for item in cur_attr]
                cur_attr = [entity_attr_template(cur_entity, item) for item in cur_attr]
                original_cur_attr_lst += cur_attr

                cur_cluster = [item['attribute'].replace(' | ', ', ') for item in step_value]
                for j, item in enumerate(cur_cluster):
                    if ',' not in item:
                        cur_cluster[j] = entity_attr_template(cur_entity, item)
                    if ',' in item:
                        cur_cluster[j] = ', '.join([entity_attr_template(cur_entity, attr) for attr in item.split(',')])
        
        cur_attr_lst = ', '.join(list(set(original_cur_attr_lst)))

        cur_prompt = ast.literal_eval(content_template.replace('{attributes}', cur_attr_lst))[0]
        cur_input = example_prompt + [cur_prompt]

        out = chat_inference(cur_input, 'gpt-3.5-turbo')

        results[key] = {
            'original_attribute': original_cur_attr_lst,
            'input_attribute': cur_attr_lst,
            'grouped_attribute': parse_result(out),
        }

    with open(args.out_path, 'w') as f:
        json.dump(results, f, indent=4)
    f.close()

if __name__ == '__main__':
    main()
