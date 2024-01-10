# %%
import pandas as pd
import argparse
import openai
import time
from openai.error import (
    RateLimitError,
    ServiceUnavailableError,
    APIError,
    APIConnectionError,
)
import os
from tqdm import tqdm
import json
import numpy as np
import ipdb
import pathlib
import re
import tiktoken
import difflib
from split_words import Splitter

from protosp01.skillExtract.prompt_template_ss import PROMPT_TEMPLATES
from utils import *

# fmt: off
## skipping black formatting for argparse

#%%
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--datapath", type=str, help="Path to source data", default = "CVTest_final.csv")
    parser.add_argument("--datapath", type=str, help="Path to source data", default = "../data/annotated/CVTest_final.csv")
    # parser.add_argument("--taxonomy", type=str, help="Path to taxonomy file in csv format", default = "taxonomy_files/taxonomy_V3.csv")
    parser.add_argument("--taxonomy", type=str, help="Path to taxonomy file in csv format", default = "../data/taxonomy/taxonomy_V4.csv")
    parser.add_argument("--openai_key", type=str, help="openai keys", default = API_KEY)
    parser.add_argument("--model", type=str, help="Model to use for generation", default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, help="Temperature for generation", default=0.3)
    parser.add_argument("--max_tokens", type=int, help="Max tokens for generation", default=40)
    parser.add_argument("--top_p", type=float, help="Top p for generation", default=1)
    parser.add_argument("--frequency_penalty", type=float, help="Frequency penalty for generation", default=0)
    parser.add_argument("--presence_penalty", type=float, help="Presence penalty for generation", default=0)
    parser.add_argument("--output_path", type=str, help="Output for evaluation results", default="results/")
    parser.add_argument("--num-samples", type=int, help="Number of samples to evaluate", default=0)
    parser.add_argument("--do-extraction", type=bool, help="Wether to do the extraction or directly the matching", default=False)
    parser.add_argument("--do-matching", type=bool, help="Wether to do the matching or not", default=False)
    
    args = parser.parse_args()
    # fmt: on
    data_type = 'cv'

    args.api_key = API_KEY #args.openai_key
    args.output_path = args.output_path + data_type + '_' + args.model + '.json'
    print("Output path", args.output_path)

    # Load data
    cv = pd.read_csv(args.datapath, sep=";", encoding = 'utf-8')
    print("loaded data:", len(cv), "sentences")
    if args.num_samples > 0:
        cv = cv.sample(args.num_samples)
        print("sampled data:", len(cv), "sentences")

    cv_json = []
    for row in cv.iterrows():
        row_dict = {}
        row_dict["sentence"] = row[1]["Sentence"]
        row_dict["groundtruth_skills"] = []
        extracted_elements = [row[1]["Extracted Element 1"], row[1]["Extracted Element 2"], row[1]["Extracted Element 3"]]
        matched_elements = [row[1]["Associated Element 1"], row[1]["Associated Element 2"], row[1]["Associated Element 3"]]
        for skill, matched_skill in zip(extracted_elements, matched_elements):
            if skill not in ["None", "NaN"] and skill != np.nan:
                row_dict["groundtruth_skills"].append({skill: matched_skill})
        cv_json.append(row_dict)

    # extract skills
    if args.do_extraction:
        print("Starting extraction")
        api = OPENAI(args, cv_json)
        api.do_prediction("extraction")
        write_json(api.data, args.output_path)

    # TODO: AD update boolean argument regarding do extraction or do matching

    # load taxonomy
    taxonomy, skill_names, skill_definitions = load_taxonomy(args)

    # load extracted skills
    cv_updated = read_json(args.output_path)

    # do matching to select candidate skills from taxonomy
    splitter = Splitter()
    max_candidates = 10
    for i, sample in enumerate(cv_updated):
        sample = select_candidates_from_taxonomy(sample, taxonomy, skill_names, skill_definitions, splitter, max_candidates)
        cv_updated[i] = sample
    write_json(cv_updated, args.output_path)

    # match skills with taxonomy
    if args.do_matching:
        print("Starting matching")
        api = OPENAI(args, cv_updated)
        api.do_prediction("matching")
        write_json(api.data, args.output_path)

    # load matched skills
    cv_updated = read_json(args.output_path)

    # Do exact match with technologies, languages, certifications
    tech_certif_lang = pd.read_csv('../data/taxonomy/tech_certif_lang.csv')
    cv_updated = exact_match(cv_updated, tech_certif_lang)
    
    # Output final 
    categs = ['Technologies', 'Certifications', 'Languages']
    clean_output = {categ: [] for categ in categs}
    clean_output['skills'] = []
    for i, sample in enumerate(cv_updated):
        for cat in categs:
            clean_output[cat].extend(sample[cat])
        for skill in sample['matched_skills']:
            clean_output['skills'].append(sample['matched_skills'][skill])
    write_json(clean_output, args.output_path.replace('.json', '_clean.json'))
    print("Done")

if __name__ == "__main__":
    main()
