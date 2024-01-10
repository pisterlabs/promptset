import os
import random
import time
import json
import math
import argparse
import numpy as np
from numpy.linalg import norm
import openai
import copy
from tqdm import tqdm
from collections import defaultdict
from rank_bm25 import BM25Okapi
from database_prompt_construciton import generate_create_table_prompt, prompt_length_by_db, OOD_SCHEMA_MAXLEN
from sql_generation import *

from sql_processing import find_random_examples, find_simsql, find_covsql


def zero_shot(dataset, model="codex", prompt_db="CreateTableSelectCol", save_prompt_only=False):
    output_path = f"outputs/{model}/{dataset}/zero_shot"
    dataset
   
    db_ids = db_ids_dataset[dataset]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    prompts_total = []
    predictions_total = []
    questions_by_db = defaultdict(list)
    with open(f"data/{dataset}/questions/questions.json", "r") as f:
        questions = json.load(f)
    for q in questions:
        questions_by_db[q["db_id"]].append(q)

    for db_id in db_ids:
        print("=" * 10 + db_id + "=" * 10)
        db_prompt = generate_create_table_prompt(dataset, db_id, prompt_db=prompt_db)
        questions = questions_by_db[db_id]
        prompts, predictions = text_to_sql_direct(model, questions, db_prompt, save_prompt_only)
        prompts_total.extend(prompts)
        if not save_prompt_only:
            predictions_total.extend(predictions)

    with open(f"{output_path}/input_prompts.json", "w") as f:
        json.dump(prompts_total, f, indent=4)
    if save_prompt_only:
        return
    
    with open(f"{output_path}/pred.json", "w") as f:
        json.dump(predictions_total, f, indent=4)
    
    with open(f"{output_path}/pred.sql", "w") as f:
        for d in predictions_total:
            f.write(d["predicted_sql"].replace('\n', ' ') + '\t' + d["db_id"] + '\n')
    


def few_shot_inoutdomain(setting, dataset,  model, prompt_db, num_table=4, num_shot_per_table=5, num_shot=5, indomain_retrieval_strategy="random",
                         outdomain_retrieval_strategy="random",  example_correctness="all",
                         synthetic_data=None, split="template", deduplicate_demo="template",
                         seed=12345, save_prompt_only=False):

    dataset_for_input = dataset

    if dataset.startswith("spider_"):
        dataset_for_input = "spider"
    if dataset.startswith("kaggle-dbqa_"):
        dataset_for_input = "kaggle-dbqa"
    db_ids = db_ids_dataset[dataset_for_input]
    output_path = f"outputs/{model}/{dataset}/few_shot_{setting}"
    if setting == "indomain":
        shot_name = f"indomain_shot_{num_shot}"
    elif setting == "outdomain":
        shot_name = f"outdomain_table_{num_table}_shot_{num_shot_per_table}"
    elif setting == "inoutdomain":
        shot_name = f"outdomain_table_{num_table}_shot_{num_shot_per_table}_indomain_shot_{num_shot}"
    else:
        raise "unknown setting"

    if setting in ["outdomain", "inoutdomain"]:
        output_path += f"_{outdomain_retrieval_strategy}"
        output_path += f"_db_len_{OOD_SCHEMA_MAXLEN}"
    if setting in ["indomain", "inoutdomain"]:
        output_path += f"_{indomain_retrieval_strategy}"
        if synthetic_data:
            output_path += f"_{synthetic_data}"
        if example_correctness != "all":
            output_path += f"_{example_correctness}"

    if indomain_retrieval_strategy == "random" or outdomain_retrieval_strategy == "random":
        output_path = output_path + f"_seed{seed}"
    random.seed(seed)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if setting in ["outdomain", "inoutdomain"]:
        with open(f"data/spider-train/questions/questions.json", "r") as f:
            outdomain_questions = json.load(f)
        outdomain_questions = [q for q in outdomain_questions if prompt_length_by_db[q["db_id"]] < OOD_SCHEMA_MAXLEN]
    
    few_shot_in_prompts = {}
    prompts_total = []
    predictions_total = []
    questions_per_db=defaultdict(list)
    with open(f"data/{dataset_for_input}/questions/questions.json", "r") as f:
        questions = json.load(f)
    for q in questions:
        questions_per_db[q["db_id"]].append(q)
    if synthetic_data:
        synthetic_questions_per_db = defaultdict(list)
        with open(f"data/{dataset_for_input}/questions/questions_{synthetic_data}.json", "r") as f:
            synthetic_questions = json.load(f)
        for q in synthetic_questions:
            synthetic_questions_per_db[q["db_id"]].append(q)
    for db_id in db_ids:
        questions =questions_per_db[db_id]
        if setting in ["indomain", "inoutdomain"]:
            if synthetic_data:
                indomain_questions_for_retrieval = synthetic_questions_per_db[db_id]
            # print("number of in-domain questions for retrieval", len(indomain_questions_for_retrieval))

        outdomain_createtable_schemas_per_question = []
        outdomain_demo_examples_per_question = []
        indomain_demo_examples_per_question = []
        if setting in ["outdomain", "inoutdomain"]:
            outdomain_questions = [q for q in outdomain_questions if q["db_id"] != db_id]
            if outdomain_retrieval_strategy == "simsql_pred":
                outdomain_bm25_corpus = [q["zeroshot"]["mentions"]["columns"] + q["zeroshot"]["mentions"]["keywords"] for q in outdomain_questions]
                outdomain_bm25 = BM25Okapi(outdomain_bm25_corpus)
            

        if setting in ["indomain", "inoutdomain"]:
            if indomain_retrieval_strategy == "similarsql":
                indomain_bm25_corpus = [q["gold"]["mentions"]["columns"] + q["gold"]["mentions"]["keywords"] for q in indomain_questions_for_retrieval]
                indomain_bm25 = BM25Okapi(indomain_bm25_corpus)
            elif indomain_retrieval_strategy == "simsql_pred":
                indomain_bm25_corpus = [q["zeroshot"]["mentions"]["columns"] + q["zeroshot"]["mentions"]["keywords"] for q in indomain_questions_for_retrieval]
                indomain_bm25 = BM25Okapi(indomain_bm25_corpus)
            elif indomain_retrieval_strategy == "covsql":
                indomain_bm25_corpus = [q["gold"]["mentions"]["columns"] + q["gold"]["mentions"]["keywords"] for q in indomain_questions_for_retrieval]
                indomain_bm25 = BM25Okapi(indomain_bm25_corpus)
            else:
                raise "unknown indomain retrieval strategy"

        for i in tqdm(range(len(questions))):
            q = questions[i]
            if setting in ["outdomain", "inoutdomain"]:
                # retrieve out domain examples
                if outdomain_retrieval_strategy == "random":
                    outdomain_questions_for_retrieval = find_random_examples(q, outdomain_questions, split=None, deduplicate_demo=deduplicate_demo)
                elif outdomain_retrieval_strategy =="simsql_pred":
                    outdomain_questions_for_retrieval = find_simsql(q, outdomain_bm25, outdomain_questions, outdomain_retrieval_strategy, split=None, deduplicate_demo=deduplicate_demo)
                else:
                    raise "unknown outdomain retrieval strategy"

                examples_per_db = defaultdict(list)
                outdomain_createtable_schemas = []
                outdomain_demo_examples = []
                for retrieval_q in outdomain_questions_for_retrieval:
                    if len(examples_per_db[retrieval_q["db_id"]]) >= num_shot_per_table:
                        continue
                    examples_per_db[retrieval_q["db_id"]].append(retrieval_q)
                    if len(examples_per_db[retrieval_q["db_id"]]) == num_shot_per_table:
                        outdomain_createtable_schemas.append(
                            generate_create_table_prompt("spider-train", retrieval_q["db_id"], prompt_db, limit_value=3))
                        outdomain_demo_examples.append(examples_per_db[retrieval_q["db_id"]][::-1]) # put the most similar example closest to the query
                        if len(outdomain_createtable_schemas) == num_table:
                            outdomain_createtable_schemas = outdomain_createtable_schemas[::-1]
                            outdomain_demo_examples = outdomain_demo_examples[::-1]
                            break
                outdomain_createtable_schemas_per_question.append(outdomain_createtable_schemas)
                outdomain_demo_examples_per_question.append(outdomain_demo_examples)

            if setting in ["indomain", "inoutdomain"]:
                # retrieve in domain examples
                if indomain_retrieval_strategy == "random":
                    indomain_demo_examples = find_random_examples(q, indomain_questions_for_retrieval, split=split, deduplicate_demo=deduplicate_demo)
                    indomain_demo_examples = indomain_demo_examples[:num_shot]
                    indomain_demo_examples = indomain_demo_examples[::-1]

                elif indomain_retrieval_strategy in ["similarsql", "simsql_pred"]:
                    indomain_demo_examples = find_simsql(q, indomain_bm25, indomain_questions_for_retrieval, indomain_retrieval_strategy,
                                                                    split=split, deduplicate_demo=deduplicate_demo)
                    indomain_demo_examples = indomain_demo_examples[:num_shot]
                    indomain_demo_examples = indomain_demo_examples[::-1]
                elif indomain_retrieval_strategy == "covsql":
                    indomain_demo_examples = find_covsql(q, indomain_bm25, indomain_questions_for_retrieval, indomain_retrieval_strategy, num_shot,
                                                          split=split, deduplicate_demo=deduplicate_demo)
                    indomain_demo_examples = indomain_demo_examples[:num_shot]
                    indomain_demo_examples = indomain_demo_examples[::-1]
                else:
                    raise "unknown indomain retrieval strategy"
                
                indomain_demo_examples_per_question.append(indomain_demo_examples)
        indomain_createtable_schema = generate_create_table_prompt(dataset_for_input, db_id, prompt_db=prompt_db)

        if setting == "indomain":
            few_shot_in_prompt, prompts, predictions = text_to_sql_few_shot_indomain(model, questions, indomain_createtable_schema, indomain_demo_examples_per_question, save_prompt_only=save_prompt_only)
        elif setting == "outdomain":
            few_shot_in_prompt, prompts, predictions = text_to_sql_few_shot_outdomain(model, questions, outdomain_createtable_schemas_per_question,indomain_createtable_schema,outdomain_demo_examples_per_question, save_prompt_only=save_prompt_only)
        elif setting == "inoutdomain":
            few_shot_in_prompt, prompts, predictions = text_to_sql_few_shot_inoutdomain(model, questions, outdomain_createtable_schemas_per_question,
                                                                                        indomain_createtable_schema,outdomain_demo_examples_per_question,indomain_demo_examples_per_question,
                                                                                        save_prompt_only=save_prompt_only)
        else:
            raise "unknown setting"

        prompts_total.extend(prompts)
        predictions_total.extend(predictions)
        few_shot_in_prompts[db_id] = few_shot_in_prompt

        # with open(os.path.join(output_path, f"{db_id}_{shot_name}.json"), "w") as f:
        #     json.dump(predictions, f, indent=4)
    with open(os.path.join(output_path, f"prompts_{shot_name}.json"), "w") as f:
        json.dump(prompts_total, f, indent=4)
    if save_prompt_only:
        return 
    
    with open(os.path.join(output_path, f"pred_{shot_name}.json"), "w") as f:
        json.dump(predictions_total, f, indent=4)
    if "num_return" not in config or config["num_return"] == 1:
        with open(os.path.join(output_path, f"pred_{shot_name}.sql"), "w") as f:
            for d in predictions_total:
                f.write(d["predicted_sql"] + '\t' + d["db_id"] + '\n')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, help='setting', choices=["zeroshot", "indomain", "outdomain", "inoutdomain"],default="zero_shot")
    parser.add_argument('--dataset', default="spider", type=str, help='dataset',
                        choices=["spider",  "spider-train", "spider_synthetic_ship_codex_verified","drspider","kaggle-dbqa", "kaggle-dbqa_synthetic_ship_codex_verified" ])
    parser.add_argument('--model', type=str, default="codex", choices=["codex", "chatgpt", "chatgpt16k", "gpt4", "CodeLlama-34b-hf"])
    parser.add_argument('--prompt_db', type=str, help='prompt construction for database', default="CreateTableSelectCol")
    parser.add_argument('--retrieval_indomain', type=str, help='retrieval strategy for in-domain demonstrations', default="random")
    parser.add_argument('--retrieval_outdomain', type=str, help='retrieval strategy for out-of-domain demonstrations', default="random")
    parser.add_argument('--num_table', type=int, help='number of databases for out-of-domain demonstrations', default=4)
    parser.add_argument('--num_shot_per_table', type=int, help='number of examples per out-of-domain database', default=5)
    parser.add_argument('--num_shot', type=int, help='number of in-domain exmaples', default=5)
    parser.add_argument('--seed', type=int, help='random_seed', default=12345)
    parser.add_argument('--synthetic_data', type=str, default=None, help='what synthetic data to use')
    parser.add_argument('--save_prompt_only', action='store_true', help='only saved the input prompt instead of running text-to-sql models')
    

    args = parser.parse_args()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    dataset = args.dataset
    setting = args.setting
    model = args.model
    num_table = args.num_table
    num_shot = args.num_shot
    retrieval_indomain = args.retrieval_indomain
    retrieval_outdomain = args.retrieval_outdomain
    seed = args.seed    
    synthetic_data = args.synthetic_data
    prompt_db = args.prompt_db
    save_prompt_only = args.save_prompt_only
    if "kaggle-dbqa" in dataset:
        prompt_db=prompt_db+"_description"
    if setting == "zeroshot":
        zero_shot(dataset, model=model, prompt_db=prompt_db,save_prompt_only=save_prompt_only)
    
    elif setting == "indomain":
        retrieval_indomain="covsql"
        retrieval_outdomain = None
        num_table = None
        num_shot_per_table = None
        num_shot=5
        if synthetic_data is not None:
            split = None
            example_correctness = "correct"
        else:
            split = "template"
            example_correctness = "all"

        few_shot_inoutdomain(setting, dataset, model, prompt_db, num_table, num_shot_per_table, num_shot,
                             indomain_retrieval_strategy=retrieval_indomain, outdomain_retrieval_strategy=retrieval_outdomain,
                             example_correctness=example_correctness, split=split,
                             synthetic_data=synthetic_data,
                             seed=seed, save_prompt_only=save_prompt_only)

    elif setting in ["outdomain"]:
        retrieval_indomain = None
        retrieval_outdomain = "simsql_pred"
        split = None
        num_shot=None
        num_shot_per_table=5
        num_table=3
        few_shot_inoutdomain(setting, dataset,model, prompt_db, num_table, num_shot_per_table, num_shot=None,
                                indomain_retrieval_strategy=retrieval_indomain, outdomain_retrieval_strategy=retrieval_outdomain,
                                example_correctness="all", split=split, synthetic_data=synthetic_data, seed=seed, save_prompt_only=save_prompt_only)

    elif setting == "inoutdomain":
        retrieval_indomain="covsql"
        retrieval_outdomain = "simsql_pred"
        num_table = 4
        num_shot_per_table = 5
        num_shot = 5
        if synthetic_data:
            split = None
            example_correctness = "correct"
        else:
            split = "template"
            example_correctness = "all"
        few_shot_inoutdomain(setting, dataset,model, prompt_db, num_table, num_shot_per_table, num_shot,
                                indomain_retrieval_strategy=retrieval_indomain, outdomain_retrieval_strategy=retrieval_outdomain,
                                example_correctness=example_correctness, split=split, 
                                synthetic_data=synthetic_data, 
                                seed=seed, save_prompt_only=save_prompt_only)
