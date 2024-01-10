"""
Zero shot evaluation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from bdb import Breakpoint

import os
import argparse
import logging

import time
import numpy as np

from vision_benchmark.common.utils import log_arg_env_config, submit_predictions
from vision_benchmark.utils import comm, create_logger
from vision_benchmark.common.constants import get_dataset_hub, VISION_DATASET_STORAGE
from vision_benchmark.datasets import SimpleTokenizer, HFPTTokenizer, class_map, template_map
from vision_benchmark.evaluation import extract_features, extract_text_features, clip_zeroshot_evaluator
from vision_benchmark.config import config, update_config

import random
from tqdm import tqdm
import openai
import json
import yaml
import pathlib

from nltk.corpus import wordnet as wn
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')



import pdb

ds_list=['eurosat-clip','country211','kitti-distance','oxford-iiit-pets','ping-attack-on-titan-plus','ping-whiskey-plus','rendered-sst2','resisc45-clip','voc2007classification','caltech101','cifar10','cifar100','dtd','fer2013','fgvc-aircraft-2013b','flower102','food101','gtsrb','hateful-memes','mnist','patchcamelyon','stanfordcar']

def add_zero_shot_args(parser):
    parser.add_argument('--ds', required=False, help='Evaluation dataset configure file name.', type=str)
    parser.add_argument('--model', required=False, help='Clip model configure file name', type=str)

    parser.add_argument('--target', help='target of run. local or azureml', choices=['local', 'azureml'], default='local')

    parser.add_argument('--submit-predictions', help='submit predictions and model info to leaderboard.', default=False, action='store_true')
    parser.add_argument('--submit-by', help='Person who submits the results.', type=str)
    parser.add_argument('--save-feature', help='save feature or not.', default=False, action='store_true')
    

    # object detection (od) dataset anotation file path
    parser.add_argument('--od_path', help='the path to wikitionary.', default='../load_wiki', type=str)
    parser.add_argument('--dataset_file_name', help='dataset file_name.', default='lvis', type=str)
    parser.add_argument('--odinw_path', help='.', default='/home/chunyl/project/data/od/', type=str)
    
    # Wiki dictionary path
    parser.add_argument('--wiki', help='extract wordnet knowledge using wiki.', default=False, action='store_true')
    parser.add_argument('--wiki_path', help='the path to wikitionary.', default='../load_wiki', type=str)

    # WordNet
    parser.add_argument('--wordnet', help='extract wordnet knowledge using wordnet.', default=False, action='store_true')

    # GPT-3 
    parser.add_argument('--gpt3', help='extract gpt3 knowledge.', default=False, action='store_true')
    parser.add_argument('--max_num_questions', default="5", type=int, help='Max number of question to ask GPT-3. note that -1 indicate the entire dataset')
    parser.add_argument('--apikey', type=str, help='api key; openaiapi@microsoft.com')
    parser.add_argument('--max_tokens', type=int, default=50, help="number of max_tokens to step in the decoding stage")
    parser.add_argument('--num_dpr_contexts', type=int, default=1, help="the nubmer of dpr retrieved context to use")
    parser.add_argument('--n_shot', type=int, default=1, help="number of shots")
    parser.add_argument('--n_ensemble', type=int, default=1, help="number of ensemble")

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)    

def _construct_command(args):
    """Build a commandline from the parsed arguments"""

    cmd = ['vb_zero_shot_eval', '--ds', args.ds, '--model', args.model, '--target', args.target]

    if args.submit_predictions:
        assert args.submit_by
        cmd.extend(['--submit-predictions', '--submit-by', args.submit_by])

    return cmd


## guarantee to get prediction
def random_query(name_test, name2wiki_exist, args):

    for ni in range(args.n_shot):
        train_item_selected = name2wiki_exist[random.randint(0,len(name2wiki_exist)-1)]
        prompt += 'Q: %s\nA: %s\n\n===\n'%( train_item_selected[0], train_item_selected[1] )
    prompt += 'Q: %s\nA:'%name_test
    
    error_count,response = 0, None

    while True:
        try:
            response = openai.Completion.create(
              engine="davinci-msft",
              prompt=prompt,
              max_tokens=5,
              logprobs=1,
              temperature=0.,
              stream=False,
              stop=["\n", "<|endoftext|>"]
            )
        except:
            time.sleep(60)
            continue
        assert(response is not None)
        return response

def ask_gpt3(args, name_test, name2wiki_exist):

    repeat_count = args.n_ensemble
    n_shot = min(args.n_shot, len(name2wiki_exist))
    
    pred_answer_list, pred_prob_list = [], []
    context_key_list = None
    # context_key_list = get_context_keys(key, metric=args.sample_select,n=n_shot*repeat_count,valid_keys=traincontext_question_dict.keys())

    for repeat in range(repeat_count):
        prompt = 'Please explain the concept according to the context.\n===\n'

        for ni in range(n_shot):
            train_item_selected = name2wiki_exist[random.randint(0,len(name2wiki_exist)-1)]
            prompt += 'Q: %s\nA: %s\n\n===\n'%( train_item_selected[0], train_item_selected[1] )
        prompt += 'Q: %s\nA:'%name_test

        error_count, response = 0, None

        # print(f'Prompt: {prompt}')
        while True:
            try:
                response = openai.Completion.create(
                  engine="davinci-msft",
                  prompt=prompt,
                  max_tokens=args.max_tokens,
                  logprobs=1,
                  temperature=0.,
                  stream=False,
                  stop=["\n", "<|endoftext|>"]
                )
            except Exception as e:
                ## if overlength, use half in-context examples to get the results
                if 'maximum context length is' in str(e):
                    response = random_query(name_test, name2wiki_exist, args)
                    break
                ## system could overload, sleep and re-try
                time.sleep(60)
                if error_count>3:
                    print(e)
                    break
                error_count += 1
                continue
            break
        if response is None:
            response = random_query(name_test, name2wiki_exist, args)
            
        plist = []
        for ii in range(len(response['choices'][0]['logprobs']['tokens'])):
            if response['choices'][0]['logprobs']['tokens'][ii]=='\n':
                break
            plist.append(response['choices'][0]['logprobs']['token_logprobs'][ii])
        pred_answer_list.append( response['choices'][0]["text"] )
        pred_prob_list.append(sum(plist))

    # print(f'GPT-3: {pred_answer_list} \n\n')

    return pred_answer_list, pred_prob_list

def extract_gpt3_konwledge(config, args):

    import sys, json
    sys.path.append(args.wiki_path)
    from get_description import resolve_meaning

    wikdict_fn = os.path.join(args.wiki_path, 'wik_dict.json') 
    wik_dict = json.load(open(wikdict_fn, encoding='utf-8'))
    

    class_names = class_map.get(config.DATASET.DATASET)
    if not class_names:
        hub = get_dataset_hub()
        from vision_datasets import Usages
        manifest = hub.create_dataset_manifest(VISION_DATASET_STORAGE, None, config.DATASET.DATASET, usage=Usages.TEST_PURPOSE)
        if manifest:
            class_names = manifest[0].labelmap

    
    
    start = time.time()
    print(class_names)


    # collect knowledge from wiki
    name2wiki = {}
    wiki_answer_list = []
    for classname in tqdm(class_names, f'Extracting wiki knowledge with Wikitionary.'):
        if type(classname) == list: classname = classname[0]
        wiki_knowledge_text = resolve_meaning(classname, wik_dict)
        print(f'Wiki | {classname}: {wiki_knowledge_text}')
        name2wiki[classname] = wiki_knowledge_text

        wiki_answer_dict = {}
        wiki_answer_dict['classname'] = classname
        wiki_answer_dict['wiki'] = wiki_knowledge_text
        wiki_answer_list.append(wiki_answer_dict)

    wiki_results_json_filename = 'WIKI_' + config.DATASET.DATASET + '.tsv'
    with open( os.path.join(config.OUTPUT_DIR, wiki_results_json_filename) , 'w') as json_file:
        json.dump(wiki_answer_list, json_file)
    json_file.close()



    
    count_has_wiki_knowledge = 0 
    name2wiki_exist = []
    for name, wiki in name2wiki.items():
        if wiki:
            count_has_wiki_knowledge += 1
            name2wiki_exist.append( (name, wiki) )

    print(f'The wiki knowledge coverage is {count_has_wiki_knowledge}/{len(name2wiki)} ')


    # collect knowledge from gpt3, based on wiki

    if True:
        openai.api_key = args.apikey

        gpt3_answer_list = []
        for name_query in name2wiki.keys():
            pred_answer_list, pred_prob_list = ask_gpt3(args, name_query, name2wiki_exist)
            print(f'GPT3 | {name_query}: {pred_answer_list}')
                    
            gpt3_answer_dict = {}
            gpt3_answer_dict['classname'] = name_query
            gpt3_answer_dict['gpt3'] = pred_answer_list
            gpt3_answer_list.append(gpt3_answer_dict)

        
        logging.info(f'=> GPT3 knowledge extraction duration time: {time.time() - start:.2f}')

        gpt3_results_json_filename = 'GPT3_' + config.DATASET.DATASET + '.tsv'
        with open( os.path.join(config.OUTPUT_DIR, gpt3_results_json_filename) , 'w') as json_file:
            # print(f"gpt3_answer_dict {gpt3_answer_dict}")
            json.dump(gpt3_answer_list, json_file)
        json_file.close()

    return class_names


def hypernyms_chain(concept):
    ss = wn.synsets(concept)
    hypernyms_chain = []
    # chain_list = ss.hypernym_paths()
    while len(ss) > 0:
        ss = ss[0]
        hypernyms_chain.append(ss.lemmas()[0].name() )
        # print(f'{ss.name()}, {ss.definition()}, {ss.hypernyms()}')
        ss = ss.hypernyms()
    return hypernyms_chain
    


def extract_ic_konwledge(config, args):

    import sys, json
    sys.path.append(args.wiki_path)
    from get_description import resolve_meaning

    wikdict_fn = os.path.join(args.wiki_path, 'wik_dict.json') 
    wik_dict = json.load(open(wikdict_fn, encoding='utf-8'))
    

    class_names = class_map.get(config.DATASET.DATASET)
    # pdb.set_trace()
    if not class_names:
        hub = get_dataset_hub()
        from vision_datasets import Usages
        manifest = hub.create_dataset_manifest(VISION_DATASET_STORAGE, None, config.DATASET.DATASET, usage=Usages.TEST_PURPOSE)
        if manifest:
            class_names = manifest[0].labelmap

    
    
    start = time.time()
    print(class_names)


    # collect knowledge from wiki
    name2wiki = {}
    wiki_answer_list = []
    data_output = []
    for classname in tqdm(class_names, f'Extracting wiki knowledge with Wikitionary.'):
        if type(classname) == list: classname = classname[0]
        wiki_knowledge_text = resolve_meaning(classname, wik_dict)
        print(f'Wiki | {classname}: {wiki_knowledge_text}')
        name2wiki[classname] = wiki_knowledge_text

        wiki_answer_dict = {}
        wiki_answer_dict['classname'] = classname
        wiki_answer_dict['def_wiki'] = wiki_knowledge_text
        wiki_answer_list.append(wiki_answer_dict)

        items = {}
        items['classname'] = classname
        items['def_wiki'] = wiki_knowledge_text

        try:
            # Knowledge source 2: WordNet Hierachy
            ss = wn.synsets(classname)[0]
            lemma_names = ss.lemma_names()
            items['path_wn'] = hypernyms_chain( lemma_names[0] ) 

            # Knowledge source 3: WordNet Definition
            items['def_wn'] = ss.definition()
        except:
            items['path_wn'] = ''
            items['def_wn'] = ''

        data_output.append(items)


    path_knowledge = 'resources/knowledge/external'
    pathlib.Path(path_knowledge).mkdir(parents=True, exist_ok=True)

    # wiki_results_tsv_filename = os.path.join(config.OUTPUT_DIR, args.dataset_file_name + '_wiki.tsv')
    # with open( wiki_results_tsv_filename , 'w') as json_file:
    #     json.dump(wiki_answer_list, json_file)
    # json_file.close()


    # Using a JSON string
    json_string = json.dumps(data_output)
    wiki_results_json_filename = os.path.join(path_knowledge, args.dataset_file_name + '_knowledge.tsv')
    with open(wiki_results_json_filename, 'w') as outfile:
        outfile.write(json_string)
    outfile.close()

    count_has_wiki_knowledge = 0 
    name2wiki_exist = []
    for name, wiki in name2wiki.items():
        if wiki:
            count_has_wiki_knowledge += 1
            name2wiki_exist.append( (name, wiki) )

    print(f'The wiki knowledge coverage is {count_has_wiki_knowledge}/{len(name2wiki)} ')


    return class_names



def extract_wiki_konwledge(config, args):

    import sys, json
    sys.path.append(args.wiki_path)
    from get_description import resolve_meaning

    wikdict_fn = os.path.join(args.wiki_path, 'wik_dict.json') 
    wik_dict = json.load(open(wikdict_fn, encoding='utf-8'))
    

    

    # Opening JSON file
    od_path = os.path.join(args.od_path, args.dataset_file_name + '.json')
    data = json.load(open(od_path))

    # dict_keys(['categories', 'info', 'licenses', 'images', 'annotations'])
    
    
    class_names = class_map.get(config.DATASET.DATASET)

    if not class_names:
        hub = get_dataset_hub()
        from vision_datasets import Usages
        manifest = hub.create_dataset_manifest(VISION_DATASET_STORAGE, None, config.DATASET.DATASET, usage=Usages.TEST_PURPOSE)
        if manifest:
            class_names = manifest[0].labelmap

    start = time.time()

    # each item in data['categories']: dict_keys(['image_count', 'synonyms', 'def', 'id', 'synset', 'name', 'frequency', 'instance_count'])
    # for example:
    # {'image_count': 17, 'synonyms': ['zucchini', 'courgette'], 'def': 'small cucumber-shaped vegetable marrow; typically dark green', 'id': 1203, 'synset': 'zucchini.n.02', 'name': 'zucchini', 'frequency': 'c', 'instance_count': 178}

    # collect knowledge from wiki 
    name2wiki = {}
    wiki_answer_list = []
    data_output = []
    for items in tqdm(data['categories'], f'Extracting wiki knowledge with Wikitionary.'):

        classname = items['name']
        if type(classname) == list: classname = classname[0]

        # Knowledge source 1: Wiki Definition
        try:
            wiki_knowledge_text = resolve_meaning(classname, wik_dict)
        except:
            wiki_knowledge_text = ''

        if classname == 'pajamas':
            print(f'Wiki | {classname}: {wiki_knowledge_text}')

        items['def_wiki'] = wiki_knowledge_text
        name2wiki[classname] = wiki_knowledge_text

        wiki_answer_dict = {}
        wiki_answer_dict['classname'] = classname
        wiki_answer_dict['wiki'] = wiki_knowledge_text
        wiki_answer_list.append(wiki_answer_dict)

        try:
            # Knowledge source 2: WordNet Hierachy
            ss = wn.synset(items['synset']) 
            # lemma_names = ss.lemma_names()[0]
            items['path_wn'] = hypernyms_chain( items['synonyms'][0] )  # items['synonyms']

            # Knowledge source 3: WordNet Definition
            items['def_wn'] = ss.definition()
        except:
            items['path_wn'] = ''
            items['def_wn'] = ''
        
        # pdb.set_trace()
            

        # 
        data_output.append(items)

    wiki_results_tsv_filename = os.path.join(args.od_path, args.dataset_file_name + '_wiki.tsv')
    with open( wiki_results_tsv_filename , 'w') as json_file:
        json.dump(wiki_answer_list, json_file)
    json_file.close()


    # Using a JSON string
    data['categories'] = data_output
    json_string = json.dumps(data)
    wiki_results_json_filename = os.path.join(args.od_path, args.dataset_file_name + '_knowledge.json')
    with open(wiki_results_json_filename, 'w') as outfile:
        outfile.write(json_string)
    outfile.close()
        

    
    count_has_wiki_knowledge = 0 
    name2wiki_exist = []
    for name, wiki in name2wiki.items():
        if wiki:
            count_has_wiki_knowledge += 1
            name2wiki_exist.append( (name, wiki) )

    print(f'The wiki knowledge coverage is {count_has_wiki_knowledge}/{len(name2wiki)} ')



def extract_konwledge_for_object365(config, args):

    import sys, json
    sys.path.append(args.wiki_path)
    from get_description import resolve_meaning

    wikdict_fn = os.path.join(args.wiki_path, 'wik_dict.json') 
    wik_dict = json.load(open(wikdict_fn, encoding='utf-8'))
    

    

    # Opening JSON file
    od_path = os.path.join(args.od_path, args.dataset_file_name + '.json')
    data = json.load(open(od_path))

    # pdb.set_trace()

    start = time.time()

    # each item in data['categories']: dict_keys(['image_count', 'synonyms', 'def', 'id', 'synset', 'name', 'frequency', 'instance_count'])
    # for example:
    # {'image_count': 17, 'synonyms': ['zucchini', 'courgette'], 'def': 'small cucumber-shaped vegetable marrow; typically dark green', 'id': 1203, 'synset': 'zucchini.n.02', 'name': 'zucchini', 'frequency': 'c', 'instance_count': 178}

    # collect knowledge from wiki 
    name2wiki = {}
    wiki_answer_list = []
    data_output = []
    for classname, id in tqdm(data.items(), f'Extracting wiki knowledge with Wikitionary.'):

        items = {}
        items['classname'] = classname
        items['id'] = id

        if "/" in classname: classname = classname.split("/")[0]

        # Knowledge source 1: Wiki Definition
        try:
            wiki_knowledge_text = resolve_meaning(classname, wik_dict)
        except:
            continue

        # print(f'Wiki | {classname}: {wiki_knowledge_text}')
        name2wiki[classname] = wiki_knowledge_text
        items['def_wiki'] = wiki_knowledge_text
        

        wiki_answer_dict = {}
        wiki_answer_dict['classname'] = classname
        wiki_answer_dict['wiki'] = wiki_knowledge_text
        wiki_answer_list.append(wiki_answer_dict)

        try:
            # Knowledge source 2: WordNet Hierachy
            ss = wn.synsets(classname)[0]
            lemma_names = ss.lemma_names()
            items['path_wn'] = hypernyms_chain( lemma_names[0] ) 

            # Knowledge source 3: WordNet Definition
            items['def_wn'] = ss.definition()
        except:
            items['path_wn'] = ''
            items['def_wn'] = ''
        # pdb.set_trace()
            

        # 
        data_output.append(items)

    # Using a JSON string
    json_string = json.dumps(data_output)
    wiki_results_json_filename = os.path.join(args.od_path, args.dataset_file_name + '_knowledge.json')
    with open(wiki_results_json_filename, 'w') as outfile:
        outfile.write(json_string)
    outfile.close()
        

    
    count_has_wiki_knowledge = 0 
    name2wiki_exist = []
    for name, wiki in name2wiki.items():
        if wiki:
            count_has_wiki_knowledge += 1
            name2wiki_exist.append( (name, wiki) )
        else: 
            print(f'Missing wiki knowledge: {name}')

    print(f'The wiki knowledge coverage is {count_has_wiki_knowledge}/{len(name2wiki)} ')



def extract_konwledge_for_odinw(config, args):

    import sys, json
    sys.path.append(args.wiki_path)
    from get_description import resolve_meaning

    wikdict_fn = os.path.join(args.wiki_path, 'wik_dict.json') 
    wik_dict = json.load(open(wikdict_fn, encoding='utf-8'))

    # Opening JSON file
    od_path = os.path.join('resources/datasets/od/odinw_caption', args.dataset_file_name + '.yaml') # args.od_path
    with open(od_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # pdb.set_trace()
    # if 'OVERRIDE_CATEGORY' not in yaml_cfg['DATASETS']:
    #     return

    # json_data = yaml_cfg['DATASETS']['OVERRIDE_CATEGORY']

    register_sets = yaml_cfg['DATASETS']['REGISTER']

    
    for k, v in register_sets.items(): # 'ann_file'
        if 'ann_file' in v:
            annotation_file_path = v['ann_file']
            annotation_file_path = os.path.join(args.odinw_path, annotation_file_path)
            classname_dict_list = json.load(open(annotation_file_path, encoding='utf-8'))['categories']
            break
    # pdb.set_trace()

    # json_data = json_data.replace("'", '"')
    # classname_dict_list = json.loads(json_data)
    

    

    start = time.time()

    # collect knowledge from wiki 
    name2wiki = {}
    wiki_answer_list = []
    data_output = []
    for items in tqdm(classname_dict_list, f'Extracting wiki knowledge with Wikitionary.'):

        
        # example: {'id': 1, 'name': 'airplane', 'supercategory': 'VOC'}

        classname = items['name']

        if "/" in classname: classname = classname.split("/")[0]

        # Knowledge source 1: Wiki Definition
        try:
            wiki_knowledge_text = resolve_meaning(classname, wik_dict)
        except:
            continue

        # print(f'Wiki | {classname}: {wiki_knowledge_text}')
        name2wiki[classname] = wiki_knowledge_text
        items['def_wiki'] = wiki_knowledge_text
        

        wiki_answer_dict = {}
        wiki_answer_dict['classname'] = classname
        wiki_answer_dict['wiki'] = wiki_knowledge_text
        wiki_answer_list.append(wiki_answer_dict)

        try:
            # Knowledge source 2: WordNet Hierachy
            ss = wn.synsets(classname)[0]
            lemma_names = ss.lemma_names()
            items['path_wn'] = hypernyms_chain( lemma_names[0] ) 

            # Knowledge source 3: WordNet Definition
            items['def_wn'] = ss.definition()
        except:
            items['path_wn'] = ''
            items['def_wn'] = ''
        # pdb.set_trace()

        data_output.append(items)

    yaml_cfg['DATASETS']['KNOWLEDGE'] =  json.dumps(data_output)

    # Using a JSON string
    json_string = json.dumps(data_output)
    pathlib.Path(args.od_path).mkdir(parents=True, exist_ok=True)
    wiki_results_json_filename = os.path.join(args.od_path, args.dataset_file_name + '_knowledge.json')
    with open(wiki_results_json_filename, 'w') as outfile:
        outfile.write(json_string)
    outfile.close()

    # Output a new YAML file
    odinw_path_knowledge = 'resources/datasets/od/odinw_caption_knowledge'
    pathlib.Path(odinw_path_knowledge).mkdir(parents=True, exist_ok=True)
    wiki_results_yaml_filename = os.path.join(odinw_path_knowledge, args.dataset_file_name + '.yaml')


    with open(wiki_results_yaml_filename, 'w') as outfile:
        outputs = yaml.dump(yaml_cfg, outfile)
    outfile.close()

    count_has_wiki_knowledge = 0 
    name2wiki_exist = []
    for name, wiki in name2wiki.items():
        if wiki:
            count_has_wiki_knowledge += 1
            name2wiki_exist.append( (name, wiki) )
        else: 
            print(f'Missing wiki knowledge: {name}')

    print(f'The wiki knowledge coverage is {count_has_wiki_knowledge}/{len(name2wiki)} ')




def extract_konwledge_for_odinw_from_list(config, args):

    import sys, json
    sys.path.append(args.wiki_path)
    from get_description import resolve_meaning

    wikdict_fn = os.path.join(args.wiki_path, 'wik_dict.json') 
    wik_dict = json.load(open(wikdict_fn, encoding='utf-8'))

    classname_dict_list = ['Ambulance', 'Bus', 'Car', 'Crab', 'Lobster', 'Motorcycle', 'Shrimp', 'Truck', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dock', 'dog', 'fish', 'flat mushroom', 'hand', 'hole', 'horse', 'jellyfish', 'jetski', 'lift', 'motorbike', 'package', 'penguin', 'person', 'pistol', 'potted plant', 'puffin', 'rabbit', 'raccoon', 'shark', 'sheep', 'sofa', 'starfish', 'stingray', 'train', 'tv monitor', 'yellow mushroom']
    
    start = time.time()

    # collect knowledge from wiki 
    name2wiki = {}
    wiki_answer_list = []
    data_output = []
    for classname in tqdm(classname_dict_list, f'Extracting wiki knowledge with Wikitionary.'):

        
        # example: {'id': 1, 'name': 'airplane', 'supercategory': 'VOC'}

        items = {}

        items['name'] = classname

        if "/" in classname: classname = classname.split("/")[0]

        # Knowledge source 1: Wiki Definition
        try:
            wiki_knowledge_text = resolve_meaning(classname, wik_dict)
        except:
            continue

        # print(f'Wiki | {classname}: {wiki_knowledge_text}')
        name2wiki[classname] = wiki_knowledge_text
        items['def_wiki'] = wiki_knowledge_text
        

        wiki_answer_dict = {}
        wiki_answer_dict['classname'] = classname
        wiki_answer_dict['wiki'] = wiki_knowledge_text
        wiki_answer_list.append(wiki_answer_dict)

        try:
            # Knowledge source 2: WordNet Hierachy
            ss = wn.synsets(classname)[0]
            lemma_names = ss.lemma_names()
            items['path_wn'] = hypernyms_chain( lemma_names[0] ) 

            # Knowledge source 3: WordNet Definition
            items['def_wn'] = ss.definition()
        except:
            items['path_wn'] = ''
            items['def_wn'] = ''
        # pdb.set_trace()

        data_output.append(items)


    # Using a JSON string
    json_string = json.dumps(data_output)
    pathlib.Path(args.od_path).mkdir(parents=True, exist_ok=True)
    wiki_results_json_filename = os.path.join(args.od_path, args.dataset_file_name + '_knowledge.json')
    with open(wiki_results_json_filename, 'w') as outfile:
        outfile.write(json_string)
    outfile.close()
    print(f'Save konwledge results for odinw at: {wiki_results_json_filename}')

    count_has_wiki_knowledge = 0 
    name2wiki_exist = []
    for name, wiki in name2wiki.items():
        if wiki:
            count_has_wiki_knowledge += 1
            name2wiki_exist.append( (name, wiki) )
        else: 
            print(f'Missing wiki knowledge: {name}')

    print(f'The wiki knowledge coverage is {count_has_wiki_knowledge}/{len(name2wiki)} ')



def extract_konwledge_for_imagenet21k(config, args):


    import spacy
    nlp = spacy.load("en_core_web_sm")

    from collections import Counter

    import sys, json
    sys.path.append(args.wiki_path)
    from get_description import resolve_meaning

    wikdict_fn = os.path.join(args.wiki_path, 'wik_dict.json') 
    wik_dict = json.load(open(wikdict_fn, encoding='utf-8'))



    stat_metafile_path = '/home/chunyl/azure_mount/chunyleu_output/exp_knowledge/data_analysis/'
    args.od_path = stat_metafile_path
    dataset = 'imagenet21k'
    dataset_path  = os.path.join(stat_metafile_path, dataset + '/stats_overall/vocab_txt.tsv')

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    vocab, concept = data['vocab'], data['concept']

    classname_dict_list = concept

    start = time.time()

    # collect knowledge from wiki 
    name2wiki = {}
    wiki_answer_list = []
    data_output = []
    concept = []

    count = 0
    for classname in tqdm(classname_dict_list, f'Extracting wiki knowledge with Wikitionary.'):

        
        # example: {'id': 1, 'name': 'airplane', 'supercategory': 'VOC'}

        items = {}

        items['name'] = classname

        if "/" in classname: classname = classname.split("/")[0]

        # Knowledge source 1: Wiki Definition
        try:
            wiki_knowledge_text = resolve_meaning(classname, wik_dict)
        except:
            continue

        # print(f'Wiki | {classname}: {wiki_knowledge_text}')
        name2wiki[classname] = wiki_knowledge_text
        items['def_wiki'] = wiki_knowledge_text

        # extract entity/concept in wiki defnition
        if wiki_knowledge_text and len(wiki_knowledge_text) > 0:
            doc = nlp(wiki_knowledge_text)
            
            item_concept_list = []
            for np in doc.noun_chunks:
                # if np.root.lemma_ == '-PRON-':
                #     continue
                entity = np.text
                item_concept_list.append(entity)
            concept += item_concept_list
            items['def_wiki_entities'] = item_concept_list
        else:
            items['def_wiki_entities'] = ['']
        

        wiki_answer_dict = {}
        wiki_answer_dict['classname'] = classname
        wiki_answer_dict['wiki'] = wiki_knowledge_text
        wiki_answer_list.append(wiki_answer_dict)

        try:
            # Knowledge source 2: WordNet Hierachy
            ss = wn.synsets(classname)[0]
            lemma_names = ss.lemma_names()
            items['path_wn'] = hypernyms_chain( lemma_names[0] ) 

            # Knowledge source 3: WordNet Definition
            items['def_wn'] = ss.definition()
        except:
            items['path_wn'] = ''
            items['def_wn'] = ''
        # pdb.set_trace()

        data_output.append(items)

        # if count > 3:
        #     break

        count += 1 

        # pdb.set_trace()


    # Using a JSON string
    

    json_string = json.dumps(data_output)
    pathlib.Path(args.od_path).mkdir(parents=True, exist_ok=True)
    wiki_results_json_filename = os.path.join(args.od_path, dataset + '_knowledge.json')
    with open(wiki_results_json_filename, 'w') as outfile:
        outfile.write(json_string)
    outfile.close()
    print(f'Save konwledge results for odinw at: {wiki_results_json_filename}')


    concept_dict = Counter(concept)
    json_string = json.dumps(concept_dict)
    wiki_results_json_filename = os.path.join(args.od_path, dataset + '_concept_stats.json')
    with open(wiki_results_json_filename, 'w') as outfile:
        outfile.write(json_string)
    outfile.close()
    print(f'Save konwledge results for odinw at: {wiki_results_json_filename}')

    count_has_wiki_knowledge = 0 
    name2wiki_exist = []
    for name, wiki in name2wiki.items():
        if wiki:
            count_has_wiki_knowledge += 1
            name2wiki_exist.append( (name, wiki) )
        else: 
            print(f'Missing wiki knowledge: {name}')

    print(f'The wiki knowledge coverage is {count_has_wiki_knowledge}/{len(name2wiki)} ')



def load_or_extract_features(args, cfg):
    if cfg.MODEL.SPEC.TEXT.TOKENIZER == 'clip':
        tokenizer = SimpleTokenizer()
    elif 'hf_' in cfg.MODEL.SPEC.TEXT.TOKENIZER:
        tokenizer = HFPTTokenizer(pt_name=cfg.MODEL.SPEC.TEXT.TOKENIZER[3:])
    else:
        tokenizer = None

    # Load or extract image features.
    feature_file = os.path.join(cfg.DATASET.ROOT, 'zeroshot_features_' + cfg.MODEL.NAME.replace('/', '') + '.npy')
    if os.path.exists(feature_file):
        logging.info('Loading features from existing files.')
        with open(feature_file, 'rb') as fread:
            image_features = np.load(fread)
            text_features = np.load(fread)
            image_labels = np.load(fread)
    else:
        image_features, image_labels = extract_features(cfg, test_split_only=True)
        text_features = extract_text_features(cfg, tokenizer)
        if args.save_feature:
            logging.info('Saving features to a file.')
            with open(feature_file, 'wb') as fwrite:
                np.save(fwrite, image_features)
                np.save(fwrite, text_features)
                np.save(fwrite, image_labels)
    logging.info(f'Test size is {image_features.shape[0]}.')

    return image_features, text_features, image_labels


    


def main():

    parser = argparse.ArgumentParser(description='Zero-shot evaluation script.')
    add_zero_shot_args(parser)
    args = parser.parse_args()

    config.defrost()
    config.NAME = ""
    config.freeze()

    if args.submit_predictions:
        assert args.submit_by

    if args.target == 'azureml':
        from vision_benchmark.common.run_aml import run_aml
        setattr(args, 'target', 'local')
        run_aml(args, _construct_command, 'zero_shot')
        return

    exp_name = 'zeroshot_eval_' + f'wiki_{config.KNOWLEDGE.WIKITIONARY.USE_DEFINITION}_gpt3_{config.KNOWLEDGE.GPT3.USE_GPT3}'
    final_output_dir = create_logger(config, exp_name)
    
    if comm.is_main_process():
        log_arg_env_config(args, config, final_output_dir)

    if args.gpt3:
        extract_gpt3_konwledge(config, args)
    # if args.worndnet:
    #     extract_worndnet_konwledge(config, args)
    if args.wiki:
        # extract_konwledge_for_object365(config, args)

        values = ['eurosat-clip','country211','kitti-distance','oxford-iiit-pets','ping-attack-on-titan-plus','ping-whiskey-plus','rendered-sst2','resisc45-clip','voc2007classification','caltech101','cifar10','cifar100','dtd','fer2013','fgvc-aircraft-2013b','flower102','food101','gtsrb','hateful-memes','mnist','patchcamelyon','stanfordcar']
        for v in values:
            args.dataset_file_name = v
            args.cfg = f'resources/datasets/{v}.yaml'
            update_config(config, args)

            extract_ic_konwledge(config, args)

        # values = 'Pascal2012.yaml,AerialMaritimeDrone_large_test.yaml,AerialMaritimeDrone_large.yaml,AerialMaritimeDrone_tiled_test.yaml,AerialMaritimeDrone_tiled.yaml,Aquarium_Aquarium_Combined.v2-raw-1024.coco_test.yaml,Aquarium_Aquarium_Combined.v2-raw-1024.coco.yaml,CottontailRabbits_test.yaml,CottontailRabbits.yaml,EgoHands_generic_test.yaml,EgoHands_generic.yaml,NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco_test.yaml,NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco.yaml,Packages_Raw_test.yaml,Packages_Raw.yaml,Raccoon_Raccoon.v2-raw.coco_test.yaml,Raccoon_Raccoon.v2-raw.coco.yaml,ShellfishOpenImages_raw_test.yaml,ShellfishOpenImages_raw.yaml,VehiclesOpenImages_416x416_test.yaml,VehiclesOpenImages_416x416.yaml,pistols_export_test.yaml,pistols_export.yaml,pothole_test.yaml,pothole.yaml,thermalDogsAndPeople_test.yaml,thermalDogsAndPeople.yaml'
        # values = [v[:-5] for v in values.split(',')]
        # for v in values:
        #     args.dataset_file_name = v
        #     extract_konwledge_for_odinw(config, args)

        # args.dataset_file_name = 'odinw_meta_classname_list.yaml'
        # extract_konwledge_for_odinw_from_list(config, args)


        # extract_konwledge_for_imagenet21k(config, args)

if __name__ == '__main__':
    main()
