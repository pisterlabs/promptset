'''
python train_data/postprocess_gpt3_tags.py \
    --gpt3-output-file /Users/vijay/Documents/code/dataset-recommendation/test_abstracts_parsed_by_gpt3_postprocessed.jsonl \
    --postprocessed-output-file /Users/vijay/Documents/code/dataset-recommendation/test_abstracts_parsed_by_gpt3_postprocessed_2.jsonl

python train_data/postprocess_gpt3_tags.py \
    --gpt3-output-file /Users/vijay/Documents/code/dataset-recommendation/test_abstracts_parsed_by_galactica.jsonl \
    --postprocessed-output-file /Users/vijay/Documents/code/dataset-recommendation/test_abstracts_parsed_by_galactica_postprocessed.jsonl

'''

import argparse
import jsonlines
import openai
import os
import time
from tqdm import tqdm

import sys
sys.path.append('..')
from utils.scrub_dataset_mentions_from_tldrs import scrub_dataset_references

parser = argparse.ArgumentParser()
parser.add_argument("--test-set-queries", default="/Users/vijay/Documents/code/dataset-recommendation/scirex_abstracts.temp")
parser.add_argument("--dataset-search-collection", default="/Users/vijay/Documents/code/dataset-recommendation/data/dataset_search_collection.jsonl")
parser.add_argument("--gpt3-output-file", default="/Users/vijay/Documents/code/dataset-recommendation/scirex_abstracts_parsed_by_gpt3.jsonl")
parser.add_argument("--postprocessed-output-file", default="/Users/vijay/Documents/code/dataset-recommendation/test_abstracts_parsed_by_gpt3_postprocessed.jsonl")


import spacy
nlp = spacy.load("en_core_web_sm")

if __name__ == "__main__":
    args = parser.parse_args()

    current_lines = list(jsonlines.open(args.gpt3_output_file))
    dataset_search_collection = list(jsonlines.open(args.dataset_search_collection))
    dataset_variants = set()
    ALLOWLIST = []
    for dataset_row in dataset_search_collection:
        dataset_names = [dataset_row['id']] + dataset_row.get("variants", [])
        for d in dataset_names:
            if len(d) > 2 and d not in ALLOWLIST:
                dataset_variants.add(d)
    dataset_variants = sorted(dataset_variants, key=lambda x: len(x))
    dataset_variant_tokens = [v.split() for v in dataset_variants]
    dataset_variant_strings = [" ".join(variant_tokens) for variant_tokens in dataset_variant_tokens]

    processed_lines_2 = []
    for line_idx, line in enumerate(current_lines):
        abstract = line["abstract"]
        gpt_summary = line["Final Summary"]
        if gpt_summary == "":
            gpt_summary = "N/A"
        processed = nlp(gpt_summary)
        gpt_sentence = list(processed.sents)[0].text

        summary_tokens = gpt_sentence.split()
        dataset_name_in_tokens = False
        matching_dataset = None

        for i, variant_tokens in enumerate(dataset_variant_tokens):
            for token_idx in range(len(summary_tokens)):
                if dataset_variant_strings[i] == " ".join(summary_tokens[token_idx:token_idx+len(variant_tokens)]):
                    dataset_name_in_tokens = True
                    matching_dataset = dataset_variants[i]
                    break
            if dataset_name_in_tokens:
                break

        if dataset_name_in_tokens:
            print(f"\nAbstract: {abstract}\n")
            print(f"Current summary at Line {line_idx+1} is\n{gpt_sentence}")
            print(f"Dataset found is {matching_dataset}")
            print("Rewrite the summary to avoid using the dataset name in the summary: ")
            gpt_sentence = input()

        line["Final Summary"] = gpt_sentence
        processed_lines_2.append(line)
    
    with open(args.postprocessed_output_file, 'w') as file:
        writer = jsonlines.Writer(file)
        writer.write_all(processed_lines_2)
        writer.close()

