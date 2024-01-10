import ast
import json
import openai
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import Counter
from transformers import GPT2Tokenizer

from utils import load_json
from openai_inference import chat_inference


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_path', type=str, required=True, help='path to save results'
    )
    parser.add_argument(
        '--api_path', type=str, required=True, help='path to the api key'
    )
    parser.add_argument(
        '--num_run', type=int, required=True, help='path to the data'
    )
    return parser.parse_args()


def load_template(template_path: str) -> str:
    return ''.join(open(template_path, 'r').readlines())


def remove_determinant(word: str) -> str:
    word = word.strip()
    if word.startswith('the '):
        return word[4:]
    elif word.startswith('a '):
        return word[2:]
    elif word.startswith('an '):
        return word[3:]
    else:
        return word
    

def plural_to_singular(word: str) -> str:
    word = word.strip()
    if not word.endswith('ss') and len(word) > 4:
        if word.endswith('ies'):
            return word[:-3] + 'y'
        if word.endswith('s'):
            return word[:-1]
        else:
            return word
    else:
        return word

    
def check_syntactic(word1: str, word2: str) -> bool:
    word1, word2 = word1.strip().lower(), word2.strip().lower()
    word1 = plural_to_singular(remove_determinant(word1))
    word2 = plural_to_singular(remove_determinant(word2))
    return word1 == word2


def parse_result(res: dict) -> list:
    content = res['content']
    content = content.split('\n')[1:]
    content = [item.replace('- ', '') for item in content]
    return content


def main():
    np.random.seed(42)
    args = get_args()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # # * mini dev set
    # data = load_json('../../data/dev_data_mini_entity.json')
    # * full dev set
    data = load_json('../../data/dev_ranked_entity.json')
    header_template = load_template('./assets/templates/entity_chat_header_v2.txt')
    content_template = load_template('./assets/templates/entity_chat_template_v2.txt')

    examples = load_json('./assets/examples/entity_chat_example.json')

    openai.api_key_path = args.api_path

    # build examples
    example_prompt = ast.literal_eval(header_template)
    for entry in examples.values():
        example_prompt += entry
    
    results = {}
    for proc_id, entry in tqdm(data.items()):

        cur_goal = entry['goal']
        cur_steps = entry['steps']

        cur_step_narratives = []
        for i, step in enumerate(cur_steps):
            if i == 0:
                cur_step_narratives.append(f'First, I {step.lower()}')
            else:
                cur_step_narratives.append(cur_step_narratives[i-1] + f' Then, I {step.lower()}')

        cur_steps = cur_step_narratives[-1]

        cur_entities = list(set(entry['entities_flattened']))

        temp_entity_dict = {ent: [ent] for ent in cur_entities}
        for i in range(len(cur_entities)):  
            candidate_lst = cur_entities[i+1:]
            for j in range(len(candidate_lst)):
                entity1, entity2 = cur_entities[i], candidate_lst[j]
                if check_syntactic(entity1, entity2):
                    try:
                        temp_entity_dict[entity1].append(entity2)
                    except:
                        continue
                    del temp_entity_dict[entity2]

        cur_goal = cur_goal.replace('"', "'")
        cur_steps = cur_steps.replace('"', "'")

        cur_template = content_template.replace('{goal}', cur_goal)  \
                                       .replace('{steps}', cur_steps)  \
                                       .replace('{entities}', 'zhanwei')  \
                                       .replace('{grouped_entities}', 'zhanwei')

        cur_template = [ast.literal_eval(item) for item in cur_template.strip().split('\n')]

        cur_entities = list(temp_entity_dict.keys())
        cur_template[2]['content'] = cur_template[2]['content'].replace('zhanwei', str(cur_entities))
        cur_input = example_prompt + cur_template[:-1]

        temp_entity_lst = []
        for _ in range(args.num_run):
            out = chat_inference(cur_input)
            cur_result = parse_result(out)
            for j, res_lst in enumerate(cur_result):
                try:
                    cur_result[j] = ast.literal_eval(res_lst)
                except:
                    if res_lst.strip():
                        if res_lst.strip()[-1] == "'":
                            res_lst = res_lst + ']'
                        elif res_lst.strip()[-1] == ",":
                            res_lst = res_lst[:-1] + ']'
                        try:
                            cur_result[j] = ast.literal_eval(res_lst)
                        except:
                            cur_result[j] = []
                    else:
                        cur_result[j] = []

            temp_entity_lst.extend(cur_result)

        temp_entity_lst = [tuple(item) for item in temp_entity_lst]
        temp_entity_count = Counter(temp_entity_lst)
        
        temp_entity_cluster = []
        for cluster, count in temp_entity_count.most_common():
            if sum([item in cur_entities for item in cluster]) == len(cluster) and cluster:
                # add cluster to the final result
                temp_entity_cluster.append(cluster)
                # remove entities in the cluster from the current entity list
                cur_entities = [item for item in cur_entities if item not in cluster]
            else:
                continue

        if cur_entities:
            temp_entity_cluster.extend(tuple([cur_entities]))

        temp_entity_cluster = [list(item) for item in temp_entity_cluster]

        gen_entity_cluster = {item[0]: item for item in temp_entity_cluster}

        # add results from syntactic cluster
        counter = 0
        new_gen_entity_cluster = deepcopy(gen_entity_cluster)
        for gen_id, gen_cluster in gen_entity_cluster.items():
            for gen_entity in gen_cluster:
                for syn_cluster in temp_entity_dict.values():
                    if gen_entity in syn_cluster and len(syn_cluster) > 1:
                        new_gen_cluster = new_gen_entity_cluster[gen_id]
                        new_gen_entity_cluster[gen_id] = list(set(new_gen_cluster + syn_cluster))

        results[proc_id] = new_gen_entity_cluster

    with open(args.out_path, 'w') as f:
        json.dump(results, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()
