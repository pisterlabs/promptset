import numpy as np
import itertools
import logging
import openai
import time
import pickle
import os
from pattern.en import pluralize
from joblib import Parallel, delayed

import inflect
p = inflect.engine()
ERROR = 3
ESTIMATED_RESPONSE_TOKENS = 8 + ERROR
PROMPT_TOKEN_INFLATION = 1.25

def generate_prompt_feature_listing(concept, feature):
    feature_words = feature.split('_')
    verb = feature_words[0] 
    if verb in ['are', 'can']:
        if concept == 'Caiman':
            feature_words.insert(1, 'caimans')
        else:    
            feature_words.insert(1, pluralize(concept).lower())
        feature_words[0] = feature_words[0].capitalize()
        feature_words.append('?')

    elif verb in ['eats', 'lay']:
        feature_words.insert(0, 'Do')
        if concept == 'Caiman':
            feature_words.insert(1, 'caimans')
        else:    
            feature_words.insert(1, pluralize(concept).lower())
        if verb == 'eats':
            feature_words.remove('eats')
            feature_words.insert(2, 'eat')
        feature_words.append('?')
    
    elif verb in ['have', 'live', 'lives']:
        feature_words.insert(0, 'Do')
        if concept == 'Caiman':
            feature_words.insert(1, 'caimans')
        else:    
            feature_words.insert(1, pluralize(concept).lower())
        # checking if the noun is singular or not. REturns false if singular
        if verb == 'lives':
            feature_words.remove('lives') 
            feature_words.insert(2, 'live')    
        if verb == 'have' and not(p.singular_noun(feature_words[-1])):
            feature_words.insert(3, 'a')
        feature_words.append('?')
    
    elif verb == 'made':
        feature_words.insert(0, 'Are')
        if concept == 'Caiman':
            feature_words.insert(1, 'caimans')
        else:    
            feature_words.insert(1, pluralize(concept).lower())
        feature_words.append('?')
    else:
        print(feature)
        print('Verb structure not implemented')
    feature_words.insert(0, 'In one word, Yes/No:')
    prompt = ' '.join(feature_words) 
    characters = len(prompt)
    return prompt, characters




def get_unique_concepts_and_features(concepts, features):
    concepts_set = list(set(concepts))
    features_set = list(set(list(features)))
    concepts_set.sort()
    features_set.sort()
    return concepts_set, features_set


def create_and_fill_concept_feature_matrix(df):
    concepts = list(df['Concept']) 
    features = list(df['Feature'])
    concepts_set, features_set = get_unique_concepts_and_features(concepts, features) 
    n_concepts = len(concepts_set)
    n_features = len(features_set)
    concept_feature_matrix = np.zeros((n_concepts, n_features))
    for concept, feature in zip(concepts, features):
        concept_idx = concepts_set.index(concept)
        feature_idx = features_set.index(feature)
        concept_feature_matrix[concept_idx, feature_idx] = 1
    return concepts_set, features_set, concept_feature_matrix 


def estimated_cost(concepts_set, features_set, concept_feature_matrix, exp_name, dataset_name):
    tokens = 0
    queries = 0
    for concept, feature in itertools.product(concepts_set, features_set):
        concept_idx = concepts_set.index(concept)
        feature_idx = features_set.index(feature)
        if concept_feature_matrix[concept_idx, feature_idx] == 0:
            _, words = generate_prompt_feature_listing(concept, feature)
            tokens += np.ceil((words + 1)/0.75)*PROMPT_TOKEN_INFLATION + ESTIMATED_RESPONSE_TOKENS
            queries += 1
    logging.info('Estimated cost of running {} on {} experiment is {}'.format(exp_name, dataset_name, tokens/1000*0.06))    
    logging.info('Total queries to be made are {}'.format(queries))


## TODO Figure out a way to make sleeping time optimal
def send_gpt_prompt(prompt, model, temperature):
    prompt_start_time = time.time()
    if model == 'ada':
        try:
            response = openai.Completion.create(
                            model="text-ada-001",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
        except:
            logging.info('Sleeping for 30 in ada')
            time.sleep(30)
            response = openai.Completion.create(
                            model="text-ada-001",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
    elif model == 'davinci':
        try:
            response = openai.Completion.create(
                            model="text-davinci-002",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
        except:
            logging.info('Sleeping for 30 in davinci')
            time.sleep(30)
            response = openai.Completion.create(
                            model="text-ada-001",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
    else:
        logging.error('Only ada and davinci implemented')
    each_prompt_time = time.time() - prompt_start_time
    return response, each_prompt_time

def prompt_gpt_feature_listing(concept, feature, prompt, tokens, answer_dict, each_prompt_api_time, model, openai_api_key, temperature):
    openai.api_key = openai_api_key
    response, each_prompt_time = send_gpt_prompt(prompt, model, temperature) 
    each_prompt_api_time.append(each_prompt_time)
    answer_dict.update({(concept, feature):(response, tokens, prompt)})

def prompt_gpt_triplet(anchor, concept1, concept2, prompt, tokens, model, each_prompt_api_time, answer_dict,openai_api_key, temperature):
    openai.api_key = openai_api_key
    response, each_prompt_time = send_gpt_prompt(prompt, model, temperature)  
    each_prompt_api_time.append(each_prompt_time)
    # print("Estimated total tokens", tokens+ESTIMATED_RESPONSE_TOKENS, '....Real tokens used', response['usage']['total_tokens'])
    answer_dict.update({(anchor, concept1, concept2,):(response, tokens, prompt)}) 
    return answer_dict



def make_gpt_prompt_batches_feat_listing(concepts_set, features_set, concept_feature_matrix, exp_name):
    logging.info('Making batches')
    batches = []
    batch = []
    total_tokens = 0
    for concept, feature in itertools.product(concepts_set, features_set):
        # was 150000, changed to 100000
        if total_tokens < 100000:
            concept_idx = concepts_set.index(concept)
            feature_idx = features_set.index(feature)
            if concept_feature_matrix[concept_idx, feature_idx] == 0:
                prompt, characters = generate_prompt_feature_listing(concept, feature)
                tokens = np.ceil((characters + 1)/4)
                batch.append([concept, feature, prompt, tokens])
                total_tokens += tokens + (ESTIMATED_RESPONSE_TOKENS)
        else:
            batches.append(batch)
            batch = [[concept, feature, prompt, tokens]]
            total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS) 
            concept_idx = concepts_set.index(concept)
            feature_idx = features_set.index(feature)
            if concept_feature_matrix[concept_idx, feature_idx] == 0:
                prompt, characters = generate_prompt_feature_listing(concept, feature)
                tokens = np.ceil((characters + 1)/4)
                batch.append([concept, feature, prompt, tokens])
                total_tokens += tokens + (ESTIMATED_RESPONSE_TOKENS)
    if len(batch) != 0:
        batches.append(batch)
    logging.info('Total batches of 150000 tokesn are {}'.format(len(batches)))
    return batches

def generate_prompt_triplet(anchor, concept1, concept2):
    # prompt = 'Keywords "{}", "{}"\nQ)Which is more similar to "{}"?\na){}\nb){}'.format(concept1, concept2, anchor, concept1, concept2)
    # prompt = 'Answer using one word "{}" or "{}". Which is more similar in meaning to "{}"?'.format(concept1, concept2, anchor)
    prompt = 'Answer using only only word - "{}" or "{}" and not "{}".Which is more similar in meaning to "{}"?'.format(concept1, concept2, anchor, anchor)
    characters = len(prompt)
    return prompt, characters

## TODO Figure out optimal condition for total_tokens
def make_gpt_prompt_batches_triplet(triplets):
    total_tokens = 0
    batches = []
    batch = []
    for triplet in triplets:
        anchor, concept1, concept2 = triplet
        prompt, characters = generate_prompt_triplet(anchor, concept1, concept2)
        tokens = np.ceil((characters + 1)/4)
        if total_tokens < 100000:
            batch.append([anchor, concept1, concept2, prompt, tokens])
            total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS)
        else:
            batches.append(batch)
            batch = [[anchor, concept1, concept2, prompt, tokens]]
            total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS) 
    if len(batch) != 0:
        batches.append(batch)
    logging.info('Total batches of 150000 tokesn are {}'.format(len(batches)))
    return batches


    
        
## TODO Figure out optimal sleeping time and n_jobs
def get_gpt_responses(batches, model, openai_api_key, exp_name, results_dir, dataset_name, temperature):
    answer_dict = {}
    each_prompt_api_time = []
    start_time = time.time()
    # import ipdb;ipdb.set_trace()
    for i, batch in enumerate(batches):
        if os.path.exists(os.path.join(results_dir, dataset_name, model +'_'+ exp_name + '_{}_{}'.format(i, temperature))):
            print(os.path.join(results_dir, dataset_name, model +'_'+ exp_name + '_{}_{}'.format(i, temperature)), 'EXISTS')
            continue
        if exp_name == 'feature_listing':
            Parallel(n_jobs=10, require='sharedmem')(delayed(prompt_gpt_feature_listing)(concept, feature, prompt, tokens, answer_dict, each_prompt_api_time, model, openai_api_key, temperature) for concept, feature, prompt, tokens in batch)
        elif exp_name == 'triplet':
            Parallel(n_jobs=10, require='sharedmem')(delayed(prompt_gpt_triplet)(anchor, concept1, concept2, prompt, tokens, model, each_prompt_api_time, answer_dict,openai_api_key, temperature) for anchor, concept1, concept2, prompt, tokens in batch)
        save_responses(answer_dict, results_dir, dataset_name, exp_name, model, i, temperature)
        if len(batches) > 1:
            time.sleep(60*2)
    exp_run_time = time.time()- start_time
    logging.info('It took {}s to run the experiment'.format(exp_run_time))
    logging.info('Each api request took {}s'.format(np.mean(each_prompt_api_time)))
    logging.info('Total time in running api concurrently is {}s'.format(np.sum(each_prompt_api_time)))
    logging.info('Speedup by parallilsation was {} x'.format((np.sum(each_prompt_api_time)- exp_run_time)/exp_run_time))
    return answer_dict


def save_responses(answer_dict, results_dir, dataset_name, exp_name, model, part, temperature):
    if not os.path.exists(os.path.join(results_dir, dataset_name)):
        os.mkdir(os.path.join(results_dir, dataset_name))
    with open(os.path.join(results_dir, dataset_name, model +'_'+ exp_name + '_{}_temperature_{}'.format(part, temperature)), 'wb') as handle:
        pickle.dump(answer_dict,handle ,  protocol=pickle.HIGHEST_PROTOCOL) 