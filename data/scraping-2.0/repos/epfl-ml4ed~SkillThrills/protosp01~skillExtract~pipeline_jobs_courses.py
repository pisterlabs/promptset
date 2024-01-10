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
import random
import pathlib
import re
import tiktoken
from transformers import AutoModel, AutoTokenizer
import difflib
from split_words import Splitter
import pickle
import datetime


# %%

from prompt_template_temp import PROMPT_TEMPLATES
from utils import *

# %%


def main():
    # fmt: off

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to source data", default = "../data/processed/job_evl_all.csv")
    parser.add_argument("--taxonomy", type=str, help="Path to taxonomy file in csv format", default = "../data/taxonomy/taxonomy_V4.csv")
    parser.add_argument("--api_key", type=str, help="openai keys", default = API_KEY)
    parser.add_argument("--model", type=str, help="Model to use for generation", default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, help="Temperature for generation", default=0)
    parser.add_argument("--max_tokens", type=int, help="Max tokens for generation", default=1000)
    parser.add_argument("--shots", type=int, help="Number of demonstrations, max = 5", default=5)
    parser.add_argument("--top_p", type=float, help="Top p for generation", default=1)
    parser.add_argument("--frequency_penalty", type=float, help="Frequency penalty for generation", default=0)
    parser.add_argument("--presence_penalty", type=float, help="Presence penalty for generation", default=0)
    parser.add_argument("--candidates_method", type=str, help="How to select candidates: rules, mixed or embeddings. Default is embeddings", default="embeddings")
    parser.add_argument("--max_candidates", type=int, help="Max number of candidates to select", default=5)
    parser.add_argument("--output_path", type=str, help="Output for evaluation results", default="results/")
    parser.add_argument("--prompt_type", type=str, help="Prompt type, from the prompt_template.py file. For now, only \"skills\", \"wlevels\", and \"wreqs\". default is wreqs.", default="wreqs")
    parser.add_argument("--num-samples", type=int, help="Last N elements to evaluate (the new ones)", default=10)
    parser.add_argument("--num-sentences", type=int, help="by how many sentences to split the corpus", default=2)
    parser.add_argument("--do-extraction", action="store_true", help="Whether to do the extraction or directly the matching")
    parser.add_argument("--do-matching", action="store_true", help="Whether to do the matching or not")
    parser.add_argument("--load-extraction", type=str, help="Path to a file with intermediate extraction results", default="")
    # parser.add_argument("--word-emb-model", type=str, help="Word embedding model to use", default="agne/jobBERT-de")
    parser.add_argument("--debug", action="store_true", help="Keep only one sentence per job offer / course to debug")
    parser.add_argument("--detailed", action="store_true", help="Generate detailed output")
    parser.add_argument("--ids", type=str, help="Path to a file with specific ids to evaluate", default=None)
    parser.add_argument("--annotate", action="store_true", help="Whether to annotate the data or not")
    parser.add_argument("--language", type=str, help="Language of the data", default="de")
    parser.add_argument("--chunks", action="store_true", help="Whether data was split into chunks or not")
    # fmt: on

    ###
    # Extraction checkpoints:
    # results/course_gpt-3.5-turbo_2sent_n10_V4231025_extraction.json
    ###

    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if "job" in args.datapath.split("/")[-1]:
        args.data_type = "job"
        print("data type:" + args.data_type)
    elif "course" in args.datapath.split("/")[-1]:
        args.data_type = "course"
        print("data type:" + args.data_type)
    elif "cv" in args.datapath.split("/")[-1]:
        args.data_type = "cv"
        print("data type:" + args.data_type)
    else:
        print("Error: Data source unknown")

    if args.language != "de":
        # append "en_" to args.data_type
        args.data_type = args.language + "_" + args.data_type

    nsent = f"_{args.num_sentences}sent"
    nsamp = f"_n{args.num_samples}"
    dt = "231025"
    tax_v = f"_{args.taxonomy.split('/')[-1].split('.')[0].split('_')[-1]}"

    if args.chunks:
        chunk = args.datapath.split("_")[-1].split(".")[0]
        print("Chunk:", chunk)
        chunk = "_" + chunk
    else:
        chunk = ""

    args.api_key = API_KEY  # args.openai_key
    args.output_path = args.output_path + args.data_type + "_" + args.model + ".json"
    print("Output path", args.output_path)

    # Intitialize pretrained word embeddings
    if args.language == "de":
        word_emb = "agne/jobBERT-de"
        # word_emb = "agne/jobGBERT"
    if args.language == "en":
        word_emb = "jjzha/jobbert-base-cased"
    print("Initializing embedding model:", word_emb)
    word_emb_model = AutoModel.from_pretrained(word_emb)
    word_emb_tokenizer = AutoTokenizer.from_pretrained(word_emb)

    emb_sh = "_rules"

    taxonomy = load_taxonomy(args)

    if args.candidates_method != "rules":
        if word_emb == "agne/jobBERT-de":
            emb_sh = "_jBd"
        elif word_emb == "agne/jobGBERT":
            emb_sh = "_jGB"
        elif word_emb == "jjzha/jobbert-base-cased":
            emb_sh = "_jbEn"

        try:
            print(f"Loading embedded taxonomy for {word_emb}")
            with open(
                f"../data/taxonomy/taxonomy{tax_v}_embeddings{emb_sh}.pkl", "rb"
            ) as f:
                emb_tax = pickle.load(f)

            # assert it's the same taxonomy
            assert (emb_tax["unique_id"] == taxonomy["unique_id"]).all()
            assert (emb_tax["name+definition"] == taxonomy["name+definition"]).all()

        except:
            print(f"Loading failed, generating embedded taxonomy for {word_emb}")
            emb_tax = embed_taxonomy(taxonomy, word_emb_model, word_emb_tokenizer)
            with open(
                f"../data/taxonomy/taxonomy{tax_v}_embeddings{emb_sh}.pkl", "wb"
            ) as f:
                pickle.dump(emb_tax, f)

    if args.candidates_method == "mixed":
        emb_sh = "_mixed"

    if args.ids is not None:
        args.num_samples = 0
        with open(args.ids, "r") as f:
            ids = f.read().splitlines()
        if "vacancies" in ids[0]:
            args.data_type = "job"
        elif "learning_opportunities" in ids[0]:
            args.data_type = "course"
        elif "resume" in ids[0]:
            args.data_type = "cv"
        ids = [int(id.split("/")[-1]) for id in ids]
        print("Evaluating only ids:", len(ids))
        args.output_path = args.output_path.replace(".json", f"_ids.json")

    print("Loading data...")
    data = pd.read_csv(args.datapath, encoding="utf-8")

    if args.language != "all" and args.ids is None:
        data = data[data["language"] == args.language]

    if args.num_samples > 0:
        data = data[-args.num_samples :]

    if args.ids is not None:
        data = data[data["id"].isin(ids)]
        data_to_save = data.copy()

        data_to_save.drop(columns="fulltext", axis=1, inplace=True)
        # save the content of the ids in a separate file
        ids_content = data_to_save.to_dict("records")
        write_json(
            ids_content,
            args.output_path.replace(
                ".json", f"{nsent}{emb_sh}{tax_v}{chunk}_content.json"
            ),
        )

    print("loaded data:", len(data), "elements")

    data = data.to_dict("records")

    # We create two files:
    # 1. results_detailed.json: contains a list of jobs/courses ids
    # each job / course has a list of sentence, each sentence has all extraction details
    # 2. results_clean.json: contains a list of jobs/courses ids
    # each job / course has only a list of skills, certifications, languages, technologies
    extraction_cost = 0
    matching_cost = 0
    detailed_results_dict = {}
    if args.load_extraction != "":
        args.do_extraction = False
        try:
            with open(args.load_extraction, "r") as f:
                detailed_results_dict = json.load(f)
        except:
            print(
                "Error: could not load intermediate extraction file. Try arg --do_extraction instead"
            )
            exit()

    for i, item in tqdm(enumerate(data)):  # item is job or course in dictionary format
        print(f"*** Processing {i+1}/{len(data)} (ID: {item['id']}) ***")

        sentences = split_sentences(item["fulltext"], language=args.language)
        # breakpoint()
        if args.debug:
            sentences = [random.choice(sentences)]
        sentences_res_list = []

        for ii in range(0, len(sentences), args.num_sentences):
            sentences_res_list.append(
                {
                    "sentence": ". ".join(sentences[ii : ii + args.num_sentences]),
                }
            )

        if len(sentences_res_list) == 0:
            continue

        if args.annotate:
            # export to csv
            df = pd.DataFrame(sentences_res_list)
            df.to_csv(
                args.output_path.replace(".json", f"{nsent}{nsamp}_annot.csv"),
                index=False,
            )

        # NOTE: this is Step 1
        # extract skills
        if args.do_extraction:
            print("Starting extraction")
            api = OPENAI(args, sentences_res_list)
            sentences_res_list, cost = api.do_prediction("extraction")
            extraction_cost += cost

        if args.load_extraction != "":
            try:
                sentences_res_list = (
                    detailed_results_dict[str(item["id"])][item["skill_type"]]
                    if args.data_type.endswith("course")
                    else detailed_results_dict[str(item["id"])]
                )
            except:
                print(
                    f"Error: could not find {str(item['id'])} in intermediate extraction file. Try arg --do_extraction instead"
                )
                exit()

        # NOTE: this is Step 2
        # select candidate skills from taxonomy -
        if args.do_matching and "extracted_skills" in sentences_res_list[0]:
            print("Starting candidate selection")
            splitter = Splitter()
            max_candidates = args.max_candidates
            for idxx, sample in enumerate(sentences_res_list):
                sample = select_candidates_from_taxonomy(
                    sample,
                    taxonomy,
                    splitter,
                    word_emb_model,
                    word_emb_tokenizer,
                    max_candidates,
                    method=args.candidates_method,
                    emb_tax=None if args.candidates_method == "rules" else emb_tax,
                )
                sentences_res_list[idxx] = sample
            # breakpoint()

        # NOTE: this is Step 3
        # match skills with taxonomy
        if args.do_matching and "skill_candidates" in sentences_res_list[0]:
            print("Starting matching")
            api = OPENAI(args, sentences_res_list)
            sentences_res_list, cost = api.do_prediction("matching")
            # breakpoint()
            matching_cost += cost

        # NOTE: This is not step 4 but separate Step that is used to use string matching
        ## to find tech and certif and lang in the text

        # Do exact match with technologies, languages, certifications
        tech_certif_lang = pd.read_csv("../data/taxonomy/tech_certif_lang.csv")
        tech_alternative_names = pd.read_csv(
            "../data/taxonomy/technologies_alternative_names.csv", sep="\t"
        )
        certification_alternative_names = pd.read_csv(
            "../data/taxonomy/certifications_alternative_names.csv", sep="\t"
        )

        sentences_res_list = exact_match(
            sentences_res_list,
            tech_certif_lang,
            tech_alternative_names,
            certification_alternative_names,
            args.data_type,
        )

        # NOTE: this is the end of data collection in the pipeline, below is related to formatting output

        # TODO find a way to correctly identify even common strings (eg 'R')! (AD: look in utils exact_match)
        # Idem for finding C on top of C# and C++
        # TODO update alternative names generation to get also shortest names (eg .Net, SQL etc) (Syrielle)
        if args.data_type.endswith("course"):
            skill_type = item["skill_type"]  # to acquire or prereq
            item_id = item["id"]  # number, first level of dict
            if item_id not in detailed_results_dict:
                detailed_results_dict[item_id] = {}
            if skill_type not in detailed_results_dict[item_id]:
                detailed_results_dict[item_id][skill_type] = sentences_res_list
            else:
                detailed_results_dict[item_id][skill_type].extend(sentences_res_list)
        else:
            detailed_results_dict[item["id"]] = sentences_res_list

        if i % 10 == 0:
            # save intermediate results
            write_json(
                detailed_results_dict,
                args.output_path.replace(
                    ".json", f"{nsent}{nsamp}{emb_sh}{tax_v}_intermediate.json"
                ),
            )

    if args.debug:
        args.output_path = args.output_path.replace(
            ".json", f"{nsent}{nsamp}{emb_sh}{tax_v}{chunk}_debug.json"
        )
    if args.detailed:
        detailed_results_dict_output = {
            key: remove_level_2(value) for key, value in detailed_results_dict.items()
        }
        write_json(
            detailed_results_dict_output,
            args.output_path.replace(
                ".json", f"{nsent}{nsamp}{emb_sh}{tax_v}{chunk}_detailed.json"
            ),
        )

    if args.do_extraction:
        write_json(
            detailed_results_dict,
            args.output_path.replace(
                ".json", f"{nsent}{nsamp}{dt}{chunk}_extraction.json"
            ),
        )

    # Output final
    if not args.debug:
        categs = [
            "Technologies",
            "Technologies_alternative_names",
            "Certifications",
            "Certification_alternative_names",
        ]
        if not args.data_type.endswith("course"):
            categs.append("Languages")
        clean_output_dict = {}

        if args.data_type.endswith("course"):
            for item_id, skill_type_dict in detailed_results_dict.items():
                for skill_type, detailed_res in skill_type_dict.items():
                    if item_id not in clean_output_dict:
                        clean_output_dict[item_id] = {}

                    clean_output = clean_output_dict[item_id].get(skill_type, {})
                    if not clean_output:
                        clean_output = {categ: [] for categ in categs}
                        clean_output["skills"] = []

                    for ii, sample in enumerate(detailed_res):
                        for cat in categs:
                            clean_output[cat].extend(sample[cat])

                        if "matched_skills" in sample:
                            for skill in sample["matched_skills"]:
                                clean_output["skills"].append(
                                    sample["matched_skills"][skill]
                                )

                    # Update the clean_output for the current skill_type
                    clean_output_dict[item_id][skill_type] = clean_output
                    for key, value in clean_output_dict.items():
                        for kkey, vvalue in value.items():
                            clean_output_dict[key][kkey] = remove_namedef(vvalue)
        else:
            for item_id, detailed_res in detailed_results_dict.items():
                clean_output = {categ: [] for categ in categs}
                clean_output["skills"] = []

                for ii, sample in enumerate(detailed_res):
                    for cat in categs:
                        clean_output[cat].extend(sample[cat])

                    if "matched_skills" in sample:
                        for skill in sample["matched_skills"]:
                            clean_output["skills"].append(
                                sample["matched_skills"][skill]
                            )

                clean_output_dict[item_id] = clean_output
                for key, value in clean_output_dict.items():
                    clean_output_dict[key] = remove_namedef(value)

        write_json(
            clean_output_dict,
            args.output_path.replace(
                ".json", f"{nsent}{nsamp}{emb_sh}{tax_v}{chunk}_clean.json"
            ),
        )
    print("Done")
    print("Extraction cost ($):", extraction_cost)
    print("Matching cost ($):", matching_cost)
    print("Total cost ($):", extraction_cost + matching_cost)

    if args.detailed:
        print(
            "Saved detailed results in",
            args.output_path.replace(
                ".json", f"{nsent}{nsamp}{emb_sh}{tax_v}{chunk}_detailed.json"
            ),
        )
    print(
        "Saved clean results in",
        args.output_path.replace(
            ".json", f"{nsent}{nsamp}{emb_sh}{tax_v}{chunk}_clean.json"
        ),
    )


if __name__ == "__main__":
    main()
