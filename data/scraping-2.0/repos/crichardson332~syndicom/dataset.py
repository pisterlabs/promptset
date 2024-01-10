import json
import csv
import pdb
import util
import random
from tqdm import tqdm
from pprint import pprint
from random import sample, choice
from atomic import node_iterator
import numpy as np
import copy
from pathlib import Path
import openai

random.seed(2021)


# write prompts for each relation
prefix_map = {
    'IsAfter': 'Before this: ',
    'IsBefore': 'After this: ',
    'xNeed': 'Before that, PersonX needed: ',
    'xAttr': 'PersonX is seen as: ',
    'xEffect': 'As a result, PersonX will: ',
    'xReact': 'As a result, PersonX feels: ',
    'xWant': 'As a result, PersonX wants: ',
    'xIntent' : 'PersonX wanted: ',
    'HinderedBy': 'This can be hindered by: ',
    'HasSubEvent': 'This includes the event/action: ',
}

def toggle_speaker(speaker):
    if speaker == 'PersonX':
        return 'PersonY'
    else:
        return 'PersonX'

def verify_event(event):
    if '___' in event:
        return False
    
    if 'PersonX' not in event:
        return False

    return True

def common_items(list1, list2):
    return list(set(list1) & set(list2))

def gen_templates(split, num_samples=None, write_tails=False, do_confounders=True, min_turns=3, max_turns=8):
    # valid_relations = ['IsAfter', 'HasSubEvent', 'IsBefore', 'HinderedBy', 'xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent']
    valid_relations = ['IsAfter', 'HasSubEvent', 'IsBefore', 'xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent']

    # output file create/overwrite file
    outfile = f'output/templates/{split}.jsonl'
    with open(outfile, 'w') as f:
        pass

    # tails output
    if write_tails:
        tails_outfile = f'output/dataset/tails_{split}.jsonl'
        with open(tails_outfile, 'w') as f:
            pass

    if do_confounders:
        with open(f'output/confounders/negations.json', 'r') as f:
            negations = json.load(f)


    atomic_json = util.jsonify_atomic(split)

    keys = list(atomic_json.keys())
    random.shuffle(keys)
    # get random subset
    if num_samples:
        subgraphs = random.sample([*atomic_json.items()],k=num_samples)
    else:
        subgraphs = atomic_json.items()

    # now loop over subgraphs. Each one will yield one dialogue
    print(f'Generating dataset...')
    datum_id = -1
    for idx,subgraph in enumerate(tqdm(subgraphs)):
        head = subgraph[0]

        # randomize number of turns
        num_turns = random.randint(min_turns,max_turns)

        # find common set between the valid relations and relations in this subgraph
        # exit if this is not at least 3 (for 4 turns total)
        candidate_relations = sorted(subgraph[1].keys() & valid_relations)
        if len(candidate_relations) < min_turns:
            continue

        # now increment the id
        datum_id += 1
        datum = {'template': [head], 'id': datum_id}

        # reset num_turns if we have fewer valid relations to choose from
        num_turns = min(num_turns, len(candidate_relations)+1)

        # randomly choose relations
        relations = random.sample(candidate_relations, k=(num_turns-1))

        # for each relation, randomly choose a tail inference
        # for rel,tails in subgraph[1].items():
        for rel in relations:
            tails = subgraph[1][rel]

            # clean_tails = util.clean_pronouns([tails], personX='the person', personY='She')[0]
            tail = random.choice(tails)

            # append this to the template
            datum['template'].append(f'{prefix_map[rel]}{tail}')

            # write the tail out if we need
            if write_tails:
                json_out = {'text': tail}
                with open(tails_outfile, 'a') as f:
                    json.dump(json_out, f)
                    f.write('\n')

        # write positive example to file
        with open(outfile, 'a') as f:
            json.dump(datum, f)
            f.write('\n')

        # randomly choose a tail to negate and create a confounding dialogue
        if do_confounders:
            idx_confounder = random.randint(1, num_turns-1)
            datum_id += 1
            datum_confounder = copy.deepcopy(datum)
            datum_confounder['id'] = datum_id

            # get negations from database
            tail_prefix, old_tail = datum_confounder['template'][idx_confounder].split(': ')
            if old_tail in negations.keys():
                new_tail = negations[old_tail] 
                datum_confounder['template'][idx_confounder] = f'{tail_prefix}: {new_tail}'
                datum_confounder['idx_confounder'] = idx_confounder
                with open(outfile, 'a') as f:
                    json.dump(datum_confounder, f)
                    f.write('\n')


    print(f'Generating dataset...Done')

def gen_dialogues(split, start_index=0, end_index=None, mode='append'):
    # output file create/overwrite file
    outfile = f'output/dataset/test_dialogues_repairs.jsonl'
    # TODO check if the file exists. dont override
    # with open(outfile, 'w') as f:
    #     pass

    if mode == 'append':
        mode_char = 'a'
    elif mode == 'write':
        mode_char = 'w'

    # if append mode and start index is not defined, find the most recent index and start from there
    if mode == 'append':
        with open(outfile) as f:
            for line in f:
                pass
            start_index = int(json.loads(line)['id']) + 1

    with open(outfile, mode_char) as f:
        pass

    # openai api key
    api_key_file = Path('/Users/crichardson8/.gpt/api_key')
    with api_key_file.open() as f:
        key = f.readline()
        openai.api_key = key.replace('\n','')

    # get examples for GPT
    with open(f'examples/gpt_examples_annotated_15.txt', 'r') as f:
        examples_str = f.read()
    
    gpt_preamble = 'Translate the following templates into natural dialogues.\n\n'
    gpt_preamble += examples_str

    # read templates and get GPT response
    template_filename = f'output/templates/{split}.jsonl'
    with open(template_filename, 'r') as json_file:
        json_list = list(json_file)

    print(f'Generating dialogues for {split} split...')
    for json_str in tqdm(json_list[start_index:end_index]):
        datum = json.loads(json_str)
        gpt_prompt = gpt_preamble
        gpt_prompt += '\n\ntemplate:'
        for turn in datum['template']:
            gpt_prompt += f'\n    {turn}'
        gpt_prompt += '\ndialogue:'
        # gpt_prompt += '\n    PersonX: '

        # ping GPT api
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=gpt_prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['template:', 'dialogue:']
        )

        # pprint(gpt_prompt)
        datum['context'] = util.parse_gpt_response(response['choices'][0]['text'])[:-1]
        datum['response'] = {
            'valid': util.parse_gpt_response(response['choices'][0]['text'])[-1]
        }

        # write the new data to file
        with open(outfile, 'a') as f:
            json.dump(datum, f)
            f.write('\n')

    print(f'Generating dialogues for {split} split...Done')

def add_negations(split, start_index=0, end_index=None, mode='append'):
    # output file create/overwrite file
    outfile = f'output/dataset/{split}.jsonl'
    # TODO check if the file exists. dont override
    # with open(outfile, 'w') as f:
    #     pass
    if mode == 'append':
        mode_char = 'a'
    elif mode == 'write':
        mode_char = 'w'

    # if append mode and start index is not defined, find the most recent index and start from there
    if mode == 'append':
        with open(outfile) as f:
            for line in f:
                pass
            start_index = int(json.loads(line)['id']) + 1

    with open(outfile, mode_char) as f:
        pass

    # openai api key
    api_key_file = Path('/Users/crichardson8/.gpt/api_key')
    with api_key_file.open() as f:
        key = f.readline()
        openai.api_key = key.replace('\n','')

    # get examples for GPT
    with open(f'examples/dialogue_negations.txt', 'r') as f:
        examples_str = f.read()
    
    gpt_preamble = 'For each of the following statements, write the opposite or antonym of the statement.\n\n'
    gpt_preamble += examples_str

    # read templates and get GPT response
    infile = f'output/dataset/old_format/{split}_dialogues.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    print(f'Generating negations for {split} split...')
    for json_str in tqdm(json_list[start_index:end_index]):
        datum = json.loads(json_str)
        datum['context'] = datum['dialogue'][:-1]
        datum['response'] = {
            'valid': datum['dialogue'][-1],
        }

        gpt_prompt = f"{gpt_preamble}\ntext: {datum['response']['valid']}\nopposite: "

        # ping GPT api
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=gpt_prompt,
            temperature=0.7,
            max_tokens=50,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop='text:'
        )

        # pprint(gpt_prompt)
        datum['response']['invalid'] = response['choices'][0]['text'].replace('\n','')

        # remove dialogue key
        datum.pop('dialogue')

        # write the new data to file
        with open(outfile, 'a') as f:
            json.dump(datum, f)
            f.write('\n')

    print(f'Generating negations for {split} split...Done')

if __name__ == "__main__":
    # splits = ['train', 'dev', 'test']
    splits = ['test']
    # splits = ['dev', 'test']
    # splits = ['dev']
    # splits = ['test']
    for sp in splits:
        # gen_templates(sp, write_tails=True, do_confounders=False)
        # gen_templates(sp, write_tails=False, do_confounders=True)
        # gen_templates(sp, write_tails=False, do_confounders=False)
        # gen_dialogues(sp, start_index=8)
        # add_negations(sp, start_index=8)
        # add_negations(sp, start_index=4995, end_index=5000)
        # add_negations(sp, mode='append')
        gen_dialogues(sp, start_index=917, end_index=918, mode='write')