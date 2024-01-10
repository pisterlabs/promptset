""" Script to score {'agent', 'patient'} given the:
-noun
-associated intransitive/middle ("This [noun] [verb]+s [adverb]")
-transitive sentence with noun in subj position ("This [noun] [verb]+s this [noun] [adverb]")
-transitive sentence with noun in obj position ("Something [verb]+s this [noun] [adverb]") 
Generates a .csv file for each of the four above
Outputs: score_folder/model_name-model_size_folder/{nouns, intr, trans-subj, trans-obj}.csv

IMPORTANT: This script charges your openAI account! Use with caution.
"""

import argparse
import os
import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import json
from retry import retry

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="prompts.json")
    parser.add_argument("--example_folder", type=str)
    parser.add_argument("--score_folder", type=str, default="scores/")
    parser.add_argument("--model_name", type=str, default="ada")
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

def format_data(data_path):
    with open (data_path, 'r') as f:
        prompts = json.load(f)
    templates = []
    unique_nouns_and_labels = []
    id = 0
    for prompt_template in prompts:
        verb = prompt_template['verb']
        for adverb in prompt_template['adverbs']:
            for noun_and_label in prompt_template['nouns_and_labels']:
                noun = noun_and_label[0]
                theta = noun_and_label[1]
                template = {
                    'id': id,
                    'verb': verb,
                    'noun': noun,
                    'adverb': adverb,
                    'theta': theta
                    }
                templates.append(template)
                if noun_and_label not in unique_nouns_and_labels:
                    unique_nouns_and_labels.append(noun_and_label)
                id += 1
    return templates, unique_nouns_and_labels

@retry(Exception, delay=3, tries=-1)
def gpt3_score(model_name, prefix):
    ll = []
    for theta in ['agent', 'patient']:
        prompt = prefix + theta
        completion = openai.Completion.create(engine=model_name, prompt=prompt,
                                                          max_tokens=0,
                                                          temperature=0.0,
                                                          logprobs=0,
                                                          echo=True,
                                                          n=1)
        # "agent" and "patient" are both one token
        logprobs = completion['choices'][0]['logprobs']
        res = {k: logprobs[k] for k in ('token_logprobs', 'tokens')}
        theta_logprob = res['token_logprobs'][-1]
        ll.append(theta_logprob)
    pred = 'agent'
    ll_agent = ll[0]
    ll_patient = ll[1]
    if ll_patient > ll_agent:
        pred = 'patient'
    return ll_agent, ll_patient, pred

def score_nouns(example_folder, nouns_and_labels, model_name):
    df = pd.DataFrame(columns=['noun', 'LL_agent', 'LL_patient', 'semantic role', 'predicted'])
    with open(example_folder + 'noun_examples.txt', 'r') as file:
        examples = file.read() + '\n'
    for noun_and_label in tqdm(nouns_and_labels):
        noun = noun_and_label[0]
        label = noun_and_label[1]
        noun_prefix = "noun: " + noun + "\n" + "agent/patient: "
        prefix = examples + noun_prefix
        ll_agent, ll_patient, prediction = gpt3_score(model_name, prefix)
        row = pd.DataFrame({
            'noun': noun,
            'LL_agent': ll_agent,
            'LL_patient': ll_patient,
            'semantic role': label,
            'predicted': prediction}, index=[0])
        df = pd.concat([row,df])    
    return df

def score_intr(example_folder, data, model_name):
    df = pd.DataFrame(columns=['id', 'verb', 'noun', 'adverb', 'sentence', 'LL_agent', 'LL_patient', 'semantic role', 'predicted'])
    with open(example_folder + 'intr_examples.txt', 'r') as file:
        examples = file.read() + '\n'
    for template in tqdm(data):
        sentence = "Sentence: This " + template['noun'] + " " + template['verb'] + " " + template['adverb'] + ".\n"
        prefix = examples + sentence + "Is " + template['noun'] + " an agent or a patient?: "
        ll_agent, ll_patient, pred = gpt3_score(model_name, prefix)
        row = pd.DataFrame({
            'id': template['id'],
            'verb': template['verb'],
            'noun': template['noun'],
            'adverb': template['adverb'],
            'sentence': "This " + template['noun'] + " " + template['verb'] + " " + template['adverb'] + ".",
            'LL_agent': ll_agent,
            'LL_patient': ll_patient,
            'semantic role': template['theta'],
            'predicted': pred}, index=[0])
        df = pd.concat([row,df])    
    return df

def score_trans_subj(example_folder, data, model_name):
    df = pd.DataFrame(columns=['id', 'verb', 'noun', 'adverb', 'sentence', 'LL_agent', 'LL_patient', 'semantic role', 'predicted'])
    with open(example_folder + 'trans_examples.txt', 'r') as file:
        examples = file.read() + '\n'
    for template in tqdm(data):
        sentence = "Sentence: This " + template['noun'] + " " + template['verb'] + " something " + template['adverb'] + ".\n"
        prefix = examples + sentence + "Is " + template['noun'] + " an agent or a patient?: "
        ll_agent, ll_patient, pred = gpt3_score(model_name, prefix)
        row = pd.DataFrame({
            'id': template['id'],
            'verb': template['verb'],
            'noun': template['noun'],
            'adverb': template['adverb'],
            'sentence': "This " + template['noun'] + " " + template['verb'] + " something " + template['adverb'] + ".",
            'LL_agent': ll_agent,
            'LL_patient': ll_patient,
            'semantic role': "agent",
            'predicted': pred}, index=[0])
        df = pd.concat([row,df])    
    return df

def score_trans_obj(example_folder, data, model_name):
    df = pd.DataFrame(columns=['id', 'verb', 'noun', 'adverb', 'sentence', 'LL_agent', 'LL_patient', 'semantic role', 'predicted'])
    with open(example_folder + 'trans_examples.txt', 'r') as file:
        examples = file.read() + '\n'
    for template in tqdm(data):
        sentence = "Sentence: Something " + template['verb'] + " this " + template['noun'] + " " + template['adverb'] + ".\n"
        prefix = examples + sentence + "Is " + template['noun'] + " an agent or a patient?: "
        ll_agent, ll_patient, pred = gpt3_score(model_name, prefix)
        row = pd.DataFrame({
            'id': template['id'],
            'verb': template['verb'],
            'noun': template['noun'],
            'adverb': template['adverb'],
            'sentence': "Something " + template['verb'] + " this " + template['noun'] + " " + template['adverb'] + ".",
            'LL_agent': ll_agent,
            'LL_patient': ll_patient,
            'semantic role': "patient",
            'predicted': pred}, index=[0])
        df = pd.concat([row,df])    
    return df

def main():
    openai.api_key = open("openAI_api.key").read()
    args = get_args()
    data_path = args.data_path
    model = args.model_name
    example_folder = args.example_folder
    templates, unique_nouns_and_labels = format_data(data_path)
    
    print("scoring nouns: ")
    noun_scores = score_nouns(example_folder, unique_nouns_and_labels, model)
    noun_scores.to_csv(args.score_folder + "/noun_scores/" + model + ".csv", index=False)
    print("scoring intr: ")
    intr_scores = score_intr(example_folder, templates, model)
    intr_scores.to_csv(args.score_folder + "/intr_scores/" + model + ".csv", index=False)
    print("scoring trans: ")
    trans_subj_scores = score_trans_subj(example_folder, templates, model)
    trans_subj_scores.to_csv(args.score_folder + "/trans_subj_scores/" + model + ".csv", index=False)
    trans_obj_scores = score_trans_obj(example_folder, templates, model)
    trans_obj_scores.to_csv(args.score_folder + "/trans_obj_scores/" + model + ".csv", index=False)
    
if __name__ == "__main__":
    main()