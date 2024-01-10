import os 
import ast
import json
import openai
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict
from transformers import GPT2Tokenizer

from openai_inference import chat_inference


def load_json(path: str) -> List[Dict[str, str]]:
    """
    Load json file from a given path

    Args:
        path: path to the json file

    Returns:
        json file as a dictionary 
    """
    with open(path, 'r') as f:
        data = json.load(f)
    f.close()
    return data


def count_tokens(inp: str) -> int:
    """
    Count the number of tokens in a given string

    Args:
        inp: input string

    Returns:
        number of tokens
    """
    inp_tokenized = tokenizer(inp).input_ids
    return len(inp_tokenized)


def make_example(example_dict: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Make n-shot in-context example for ChatGPT

    Args:
        example_dict: dicionary of example prompts

    Returns:
        example prompt in ChatGPT format (https://platform.openai.com/docs/guides/chat/introduction)
    """
    header = example_dict[0]
    example_list = example_dict[1:]
    total_example = int(len(example_list) / 4)

    example_idx = np.random.choice(total_example, args.num_shot, replace=False)
    out = []
    total_token = 0
    for idx in example_idx:
        out.append(example_list[idx * 4: idx * 4 + 4])
        total_token += sum([count_tokens(item['content']) for item in example_list[idx: idx + 4]])

    out = [lst for sublst in out for lst in sublst]

    out = [header] + out
    total_token += count_tokens(header['content'])
    print(f'Example contains {total_token} tokens.')
    return out


def parse_result(result: dict) -> list:
    """
    Parse ChatGPT generation from a string to list of entities

    Args:
        result: the generated dictionary from ChatGPT

    Returns:
        a list of entities
    """
    res = result['content'].split('\n')[-1]
    if "'" not in res:
        return []
    else:
        # clean generation
        if res.strip()[-1] == ',':
            res = res.strip()[:-1] + ']'
        elif res.strip()[0] == '[' and res.strip()[-2:] != "']":
            res = res + "']"

        # convert generation to list
        try:
            return ast.literal_eval(res)
        except:
            return []


def get_args() -> argparse.Namespace:
    """
    Get arguments from command line

    Returns:
        arguments as a namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_shot', type=int, default=3, help='number of shots for in-context learning'
    )
    parser.add_argument(
        '--num_run', type=int, default=3, help='number of runs for each entity'
    )
    return parser.parse_args()


def main():
    data = load_json('../../../data/data_in_new_format/dev-data-reformatted-v4.json')
    entity_example = load_json('./assets/entity_example.json')
    entity_template = load_json('./assets/entity_template.json')
    openai.api_key_path = os.path.expanduser('~/harry.key')
    np.random.seed(42)

    global args, tokenizer
    args = get_args()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    entity_example = make_example(entity_example)

    for proc_id, proc_content in tqdm(data.items()):
        cur_goal = proc_content['goal']
        cur_steps = proc_content['steps']
        cur_clusters = proc_content['clusters']
        
        cur_step_narratives = []
        for i, step in enumerate(cur_steps):
            if i == 0:
                cur_step_narratives.append(f'First, I {step.lower()}')
            else:
                cur_step_narratives.append(cur_step_narratives[i-1] + f' Then, I {step.lower()}')

        cur_goal = f'I am trying to {cur_goal.strip().lower()}. '
        cur_steps = cur_goal + cur_step_narratives[-1]

        # over generate the cluster entries
        for entity_id, entity_content in cur_clusters.items():
            entity_cluster = entity_content['entity_cluster']
            cur_template = deepcopy(entity_template)

            cur_context = cur_template[0]['content'].replace('{context}', cur_steps) + ' Do you get the procedure?'
            cur_template[0]['content'] = cur_context
            cur_entity_context = f'Here are a group of entity names describing the same object: {entity_cluster}.'
            cur_entity_context = cur_template[2]['content'].replace('{entity_context}', cur_entity_context)
            cur_entity_context += ' What are some alternative names for this object? Organize them in a list. Answer \"None\" if there is no alternative name.'
            cur_template[2]['content'] = cur_entity_context

            cur_template = cur_template[:-1]

            cur_input = entity_example + cur_template

            all_gen_cluster = []
            for _ in range(args.num_run):
                out = chat_inference(cur_input)
                gen_cluster = parse_result(out)
                all_gen_cluster.extend(gen_cluster)

            data[proc_id]['clusters'][entity_id]['entity_cluster'] += list(set(all_gen_cluster))

    with open('../../../data/data_in_new_format/dev-data-reformatted-v4-entity-overgenerated.json', 'w') as f:
        json.dump(data, f, indent=4)
    f.close()

if __name__ == "__main__":
    main()

