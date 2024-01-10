import os
import ast
import json
import openai
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from collections import Counter
from transformers import GPT2Tokenizer

from openai_inference import chat_inference
from utils import load_json, sample_data, save_json, load_txt, clean_steps


def fill_template(template: str, cur_goal: str, cur_steps: str, entity_attr: list, entity: str):
    entity_attr = [item.split(' | ') for item in entity_attr]
    entity_attr = [lst for sublst in entity_attr for lst in sublst]
    entity_attr = [item.strip() for item in entity_attr]

    entity = entity.split('|')[0].strip()
    
    if args.format == 'concise':
        entity_attr = [f'{item} of {entity}' for item in entity_attr]

    cur_goal = cur_goal.replace('"', "'")
    cur_steps = cur_steps.replace('"', "'")

    cur_template = template.replace('{goal}', cur_goal)  \
                            .replace('{steps}', cur_steps)  \
                            .replace('{attributes}', 'zhanwei') 

    entity_attr_str = str(entity_attr).replace('[', '').replace(']', '')

    return cur_template, entity_attr, entity_attr_str


def load_template(template_path: str) -> str:
    return ''.join(open(template_path, 'r').readlines())


def make_example(example_dict: dict):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    MAX_TOKEN = 1500
    cur_token = 0

    out_example = []
    for cur_example in example_dict.values():
        for entry in cur_example:
            cur_token += len(tokenizer.encode(entry['content']))
        
        if cur_token > MAX_TOKEN:
            break

        out_example.extend(cur_example)
    
    print(f'prompt contains {cur_token} tokens')
    return out_example


def parse_result(res: dict) -> list:
    content = res['content']
    content = content.split('\n')[1:]
    content = [item.replace('- ', '') for item in content]
    return content


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_path', type=str, required=True, help='path to save results'
    )
    parser.add_argument(
        '--api_path', type=str, required=True, help='path to the api key'
    )
    parser.add_argument(
        '--format', type=str, default='vanilla', help='format of the output, choose from ["vanilla", "concise"]'
    )
    parser.add_argument(
        '--num_run', type=int, default=1, help='number of runs'
    )
    return parser.parse_args()


def main():
    global args
    np.random.seed(42)
    args = get_args()
    # # # * mini dev set
    # data = load_json('../../data/dev_data_mini_attr.json')
    # * full dev set
    data = load_json('../../data/dev_ranked_attr.json')
    header_template = load_template('./assets/templates/attr_chat_header_v2.txt')
    template = load_template('./assets/templates/attr_chat_template_v2.txt')

    examples = load_json(f'./assets/examples/attribute_chat_{args.format}_example.json')

    openai.api_key_path = args.api_path

    # build examples
    example = make_example(examples)
    example_prompt = ast.literal_eval(header_template) + example
    
    results = {}
    for proc_id, entry in tqdm(data.items(), position=0, leave=False):
        cur_goal = entry['goal']
        cur_steps = entry['steps']

        cur_step_narratives = []
        for i, step in enumerate(cur_steps):
            if i == 0:
                cur_step_narratives.append(f'First, I {step.lower()}')
            else:
                cur_step_narratives.append(cur_step_narratives[i-1] + f' Then, I {step.lower()}')

        cur_steps = cur_step_narratives[-1]

        cur_entity_attr_dict = entry['entity_attr_dict']

        temp_entity_dict = {}
        for entity_id, entity_attr in tqdm(cur_entity_attr_dict.items(), position=1, leave=False):

            if len(entity_attr) == 1:
                temp_attr_dict = {entity_attr[0]: [entity_attr[0]]}
            else:
                cur_template, entity_attr, entity_attr_str = fill_template(template, cur_goal, cur_steps, entity_attr, entity_id)
                cur_template = ast.literal_eval(cur_template)[:-1]
                cur_template[-1]['content'] = cur_template[-1]['content'].replace('zhanwei', entity_attr_str)

                cur_input = example_prompt + cur_template

                # run ChatGPT 3 times and aggregate results
                cur_result_list = []
                for _ in range(args.num_run):
                    out = chat_inference(cur_input)
                    cur_result = parse_result(out)
                    cur_result_list.extend(cur_result)

                cur_result_count = Counter(cur_result_list)
                cur_result = []
                for cluster, count in cur_result_count.most_common():
                    cur_candidate = cluster.split(',')
                    cur_candidate = [item.strip() for item in cur_candidate]
                    if sum([item in entity_attr for item in cur_candidate]) == len(cur_candidate):
                        cur_result.append(cluster)
                        entity_attr = [item for item in entity_attr if item not in cur_candidate]
                
                if entity_attr:
                    cur_result.extend(entity_attr)

                temp_attr_dict = {}
                for res in cur_result:
                    res = res.split(',')
                    res = [item.strip() for item in res]
                    if len(res) > 1:
                        temp_attr_dict[res[0]] = res[1:]
                    else:
                        temp_attr_dict[res[0]] = [res[0]]

            temp_entity_dict[entity_id] = temp_attr_dict 

        results[proc_id] = temp_entity_dict 

    with open(args.out_path, 'w') as f:
        json.dump(results, f, indent=4)
    f.close()

if __name__ == '__main__':
    main()
