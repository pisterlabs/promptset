import json
import openai
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import os
from sentence_transformers import SentenceTransformer, util


def calculate_sentence_bert_score(d_explanations, dv_explanations):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    d_explanations_embeddings = model.encode(d_explanations, convert_to_tensor=True)
    dv_explanations_embeddings = model.encode(dv_explanations, convert_to_tensor=True)
    score = util.cos_sim(d_explanations_embeddings, dv_explanations_embeddings)
    return score.diagonal()
    
    

def calculate(args):
    with open(f"{args.root_dir}/info.json", 'r') as f:
        src_json = json.load(f)

    keys_not_to_check = [k for k, v in src_json.items() if v['dv_explanation'] == None]
    keys_to_check = [k for k, v in src_json.items() if v['dv_explanation'] != None]
    d_explanations = [v['d_explanation'] for v in src_json.values() if v['dv_explanation']!= None]
    dv_explanations = [v['dv_explanation'] for v in src_json.values() if v['dv_explanation'] != None]
    scores = calculate_sentence_bert_score(d_explanations, dv_explanations)
    
    for key in keys_not_to_check:
        src_json[key]['sentbert'] = -1.0
    for key, score in zip(keys_to_check, scores):
        src_json[key]['sentbert'] = float(score)
    
    return src_json
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_result_path', type=str, default='./pipeline_result.json', help='Directory of json file consisting of results of pipeline')
    parser.add_argument('--prompt_result_score_path', type=str, default='./pipeline_result_w_score.json', help='Directory of json file consisting of final results of pipeline')
    args = parser.parse_args()
    result = calculate(args)
    with open(args.prompt_result_score_path, 'w') as f:
        json.dump(result, f, indent=2)




    
    
    
    
    