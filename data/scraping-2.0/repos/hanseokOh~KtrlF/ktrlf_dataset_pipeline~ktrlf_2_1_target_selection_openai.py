import argparse
import pickle
from tqdm.auto import tqdm
import os
import parmap
import json
from collections import defaultdict, Counter
import logging
import asyncio
from utils.api_request_parallel_processor import process_api_requests_from_file
import time
import copy
import openai
from pathlib import Path
from utils.statistics import print_number_of_targets, print_number_of_mentions
from utils.entity_evidence import crawl_wikipedia_article


def make_entity_evidence_dict(all_task, num_evidence_sent=20):
    entity_evidence_dict = {}
    for dic in tqdm(all_task):
        for entity_dic in dic['data']['entity_info']:
            entity = entity_dic['entity']
            if entity_evidence_dict.get(entity) is not None:
                continue

            evidence = crawl_wikipedia_article(entity_dic['wikipedia_link'].split('/')[-1], num_evidence_sent)
            
            entity_evidence_dict[entity] = evidence
    return entity_evidence_dict

def _need_to_determine(qa_pair, entity, need_to_determine_key_list):
    if len(need_to_determine_key_list) == 0:
        return True
    
    _entity_cnt_dict = dict(Counter([_ent for key in need_to_determine_key_list for _ent in qa_pair[key]]))
    count = _entity_cnt_dict.get(entity)
    if count == 1:
        return True
    return False

def _dump_gpt_input(all_task, input_filepath, gpt_model_name, entity_evidence_dict, num_evidence_sent=10, need_to_determine_key_list=[]):
    system_prompt = f"""
You are a QA system to identify the given entity is the answer.
The inputs are entity, query and evidence.

You must follow this requirements.
Requirements:
- Output have to be either 'true' or 'false'
- Do not say anything except 'true' or 'false'

The example is as below.

Entity: Google
Query: Find all IT companies in Computer industry
Evidence: Google LLC (/ˈɡuːɡəl/ (listen)) is an American multinational technology company focusing on artificial intelligence,[9] online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, and consumer electronics. It has often been considered "the most powerful company in the world"[10] and as one of the world's most valuable brands due to its market dominance, data collection, and technological advantages in the field of artificial intelligence.[11][12][13] Its parent company Alphabet is often considered one of the Big Five American information technology companies, alongside Amazon, Apple, Meta, and Microsoft.
Output: true

Entity: Samsung
Query: Find all companies in United States
Evidence: Samsung Group,[3] or simply Samsung (Korean: 삼성; RR: samseong [samsʌŋ]) (stylized as SΛMSUNG), is a South Korean multinational manufacturing conglomerate headquartered in Samsung Town, Seoul, South Korea.[1] It comprises numerous affiliated businesses,[1] most of them united under the Samsung brand, and is the largest South Korean chaebol (business conglomerate). As of 2020, Samsung has the eighth highest global brand value.
Output: false
"""

    odqa_input_format = []
    for dic in all_task:
        _entity_set = dict.fromkeys([tag_dic['entity'] for tag_dic in dic['data']['entity_info']])
        for q_idx,qa_pair in enumerate(dic['data']['qa_pairs']):
            odqa_input_format += [
            {
                'id': f"{dic['id']}[SEP]q{q_idx}[SEP]e{ent_idx}",
                'question': qa_pair['question'],
                'entity': entity,
                'evidence': ' '.join(entity_evidence_dict[entity][:num_evidence_sent])
            } for ent_idx, entity in enumerate(_entity_set) if _need_to_determine(qa_pair, entity, need_to_determine_key_list)]

    all_input_format = []
    for dic in odqa_input_format:
        all_input_format.append({
            "model": gpt_model_name,
            'messages': [
                {'role': 'system', 'content': system_prompt.strip()},
                {'role': 'user', 'content': f"Entity: {dic['entity']}\nQuery: {dic['question']}\nEvidence: {dic['evidence']}\nOutput: "}
            ],
            'user': dic['id']
        })

    with open(input_filepath, 'w') as f:
        for dic in all_input_format:
            f.write(json.dumps(dic)+'\n')


def _load_gpt_output(output_filepath):
    tup_list = []
    with open(output_filepath) as f:
        for line in f:
            input_, output = json.loads(line)
            input_id = input_['user']
            try:
                generated_answer_str = output['choices'][0]['message']['content']
            except:
                generated_answer_str = ""
            tup_list.append({'id': input_id, 'output': generated_answer_str})

    # sort by original order
    generated_output_list = sorted(tup_list, key=lambda dic: dic['id'])
    return generated_output_list

def _parse_gpt_output(all_task, generated_output_list, to_answer_key):
    _output_idx_mapper = defaultdict(lambda: defaultdict(list))

    for dic in generated_output_list:
        output = dic['output'].lower().strip()
        if output != 'true':
            continue
        
        original_id, q_idx, ent_idx = dic['id'].split('[SEP]')
        q_idx = int(q_idx[1:])
        ent_idx = int(ent_idx[1:])

        _output_idx_mapper[original_id][q_idx].append(ent_idx)

    for dic in all_task:
        id = dic['id']
        _entity_set = list(dict.fromkeys([tag_dic['entity'] for tag_dic in dic['data']['entity_info']]))
        for q_idx, qa_pair in enumerate(dic['data']['qa_pairs']):
            odqa_gpt_preds = [_entity_set[ent_idx] for ent_idx in _output_idx_mapper[id][q_idx]]
            qa_pair[to_answer_key] = odqa_gpt_preds

def select_target_using_model(all_task, entity_evidence_dict, gpt_model_name, request_url, api_key, to_answer_key, need_to_determine_key_list):
    Path("./dump/.gpt_format").mkdir(parents=True, exist_ok=True)

    _timestamp = str(int(time.time()))
    input_format_path = f'./dump/.gpt_format/_openai_{_timestamp}_input_format.jsonl'
    output_format_path = f'./dump/.gpt_format/_openai_{_timestamp}_output_format.jsonl'
    max_attempts = 10

    _dump_gpt_input(all_task, input_format_path, gpt_model_name, entity_evidence_dict, num_evidence_sent=10, need_to_determine_key_list=need_to_determine_key_list)
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=input_format_path,
            save_filepath=output_format_path,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(3_000 * 0.5),
            max_tokens_per_minute=float(250_000 * 0.5),
            token_encoding_name="cl100k_base",
            max_attempts=int(max_attempts),
            logging_level=int(logging.INFO),
        )
    )
    generated_output = _load_gpt_output(output_format_path)
    _parse_gpt_output(all_task, generated_output, to_answer_key)

def clear_query_with_empty_target(all_task, answer_key):
    new_all_task = []
    for dic in all_task:
        _new_qa_pairs = [qa_dic for qa_dic in dic['data']['qa_pairs'] if len(qa_dic[answer_key])>0]
        _new_dic = copy.deepcopy(dic)
        _new_dic['data']['qa_pairs'] = _new_qa_pairs
        new_all_task.append(_new_dic)
    return new_all_task

def clear_doc_with_emtpy_query(all_task):
    new_all_task = []
    for dic in all_task:
        if len(dic['data']['qa_pairs'])==0:
            continue
        new_all_task.append(copy.deepcopy(dic))
    return new_all_task


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_evidence_cache_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--openai_model_name", type=str, choices=['gpt-3.5-turbo-0613','gpt-4-0613'])
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--openai_api_key", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    need_to_determine_key_list = ['_llama2_query_generation_preds', 'chatgpt_preds']
    to_answer_key = 'gpt4_preds'

    with open(args.input_path, 'rb') as f:
        all_task = pickle.load(f)

    # get evidence
    if args.entity_evidence_cache_path:
        with open(args.entity_evidence_cache_path, 'rb') as f:
            entity_evidence_dict = pickle.load(f)
    else:
        _num_proc = os.cpu_count()
        _batch_size = 70
        _splited_all_task = [all_task[x:x+_batch_size] for x in range(0, len(all_task), _batch_size)]
        results = parmap.map(make_entity_evidence_dict, _splited_all_task, pm_pbar=True, pm_processes=_num_proc)

        entity_evidence_dict = {k:v for dic in results for k,v in dic.items()}
        with open('./dump/entity_evidence_dict.pickle', 'wb') as f:
            pickle.dump(entity_evidence_dict, f)
 
    select_target_using_model(all_task, entity_evidence_dict, args.openai_model_name, args.openai_request_url, args.openai_api_key, to_answer_key, need_to_determine_key_list)
    print(f"[Num. of Targets] Final:")
    print(f"{print_number_of_targets(all_task, to_answer_key)}")
    
    with open(args.output_path, 'wb') as f:
        pickle.dump(all_task, f)
