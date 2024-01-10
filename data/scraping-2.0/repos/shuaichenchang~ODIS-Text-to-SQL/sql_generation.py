import os
import random
import time
import subprocess
import asyncio
import requests
import json
import math
import argparse
import numpy as np
from numpy.linalg import norm
import openai
import copy
from collections import defaultdict
from transformers import AutoTokenizer
from sql_processing import format_query

import sqlparse
import tiktoken

os.environ["DATA_GYM_CACHE_DIR"] = "~/tmp/data-gym-cache"
encoding = tiktoken.get_encoding("cl100k_base")
chatgpt_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

DB_SEP = "-- Given a relational database, generate SQLite corresponding to the question.\n"
MAX_GEN_TOKENS = 200


def get_prompt_length(prompt, model="codex"):
    if model == "codex":
        result = subprocess.run(["node", "codex_prompt_length.mjs", prompt], stdout=subprocess.PIPE)
        prompt_len = eval(result.stdout)
        return prompt_len
    elif model in ["chatgpt", "chatgpt16k", "gpt4"]:
        return len(chatgpt_encoding.encode(prompt))
    elif "llama" in model:
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-34b-Instruct-hf")
        return len(tokenizer(prompt)["input_ids"])


spider_train_db_ids = ['department_management', 'farm', 'student_assessment', 'bike_1', 'book_2', 'musical', 'twitter_1', 'product_catalog', 'flight_1',
                       'allergy_1', 'store_1', 'journal_committee', 'customers_card_transactions', 'race_track', 'coffee_shop', 'chinook_1', 'insurance_fnol',
                       'medicine_enzyme_interaction', 'university_basketball', 'phone_1', 'match_season', 'climbing', 'body_builder', 'election_representative',
                       'apartment_rentals', 'game_injury', 'soccer_1', 'performance_attendance', 'college_2', 'debate', 'insurance_and_eClaims',
                       'customers_and_invoices', 'wedding', 'theme_gallery', 'epinions_1', 'riding_club', 'gymnast', 'small_bank_1', 'browser_web', 'wrestler',
                       'school_finance', 'protein_institute', 'cinema', 'products_for_hire', 'phone_market', 'gas_company', 'party_people', 'pilot_record',
                       'cre_Doc_Control_Systems', 'company_1', 'local_govt_in_alabama', 'formula_1', 'machine_repair', 'entrepreneur', 'perpetrator', 'csu_1',
                       'candidate_poll', 'movie_1', 'county_public_safety', 'inn_1', 'local_govt_mdm', 'party_host', 'storm_record', 'election', 'news_report',
                       'restaurant_1', 'customer_deliveries', 'icfp_1', 'sakila_1', 'loan_1', 'behavior_monitoring', 'assets_maintenance', 'station_weather',
                       'college_1', 'sports_competition', 'manufacturer', 'hr_1', 'music_1', 'baseball_1', 'mountain_photos', 'program_share', 'e_learning',
                       'insurance_policies', 'hospital_1', 'ship_mission', 'student_1', 'company_employee', 'film_rank', 'cre_Doc_Tracking_DB', 'club_1',
                       'tracking_grants_for_research', 'network_2', 'decoration_competition', 'document_management', 'company_office', 'solvency_ii',
                       'entertainment_awards', 'customers_campaigns_ecommerce', 'college_3', 'department_store', 'aircraft', 'local_govt_and_lot',
                       'school_player', 'store_product', 'soccer_2', 'device', 'cre_Drama_Workshop_Groups', 'music_2', 'manufactory_1',
                       'tracking_software_problems', 'shop_membership', 'voter_2', 'products_gen_characteristics', 'swimming', 'railway',
                       'customers_and_products_contacts', 'dorm_1', 'customer_complaints', 'workshop_paper', 'tracking_share_transactions', 'cre_Theme_park',
                       'game_1', 'customers_and_addresses', 'music_4', 'roller_coaster', 'ship_1', 'city_record', 'e_government', 'school_bus',
                       'flight_company', 'cre_Docs_and_Epenses', 'scientist_1', 'wine_1', 'train_station', 'driving_school', 'activity_1', 'flight_4',
                       'tracking_orders', 'architecture', 'culture_company']

spider_dev_db_ids = ['concert_singer', 'pets_1', 'car_1', 'flight_2', 'employee_hire_evaluation', 'cre_Doc_Template_Mgt', 'course_teach', 'museum_visit',
                     'wta_1', 'battle_death', 'student_transcripts_tracking', 'tvshow', 'poker_player', 'voter_1', 'world_1', 'orchestra', 'network_1',
                     'dog_kennels', 'singer', 'real_estate_properties']
kaggle_db_ids = ['WorldSoccerDataBase', 'Pesticide', 'USWildFires', 'GeoNuclearData', 'WhatCDHipHop', 'TheHistoryofBaseball', 'StudentMathScore',
                 'GreaterManchesterCrime']

db_ids_dataset = {
    "spider-train": spider_train_db_ids,
    "spider": spider_dev_db_ids,
    "drspider": spider_dev_db_ids,
    "kaggle-dbqa": kaggle_db_ids,
}

def cut_prompt_with_max_tokens(model, prompt, max_generate_tokens=MAX_GEN_TOKENS):
    if model in ["codex", "gpt4"]:
        model_max_tokens = 8000
    elif model in ["chatgpt"]:
        model_max_tokens = 4000
    elif model in ["chatgpt16k"]:
        model_max_tokens = 16000
    elif model in ["llama34instruct", "llama7", "llama13"]:
        model_max_tokens = 8000 
    else:
        raise NotImplementedError
    prompt_len = get_prompt_length(prompt, model=model)
    cnt = 0
    while prompt_len >= model_max_tokens - max_generate_tokens:
        prompt = prompt.split(DB_SEP)
        prompt = DB_SEP.join([""] + prompt[2:])
        prompt_len = get_prompt_length(prompt, model=model)
        cnt += 1
    if cnt > 0:
        print(f"Prompt too long, skip the first {cnt} databases.")

    return prompt, prompt_len


def call_chatgpt(model, prompt, max_tokens=200, stop=[";", "Question", 'Answer', '/*']):
    if model == "chatgpt":
        api_model = "gpt-3.5-turbo-0301"
        model_max_tokens = 4000
    elif model == "chatgpt16k":
        api_model = "gpt-3.5-turbo-16k-0613"
        model_max_tokens = 16000
    elif model == "gpt4":
        api_model = "gpt-4-0613"
        model_max_tokens = 8000
    else:
        raise NotImplementedError

    while (True):
        try:
            response = openai.ChatCompletion.create(
                model=api_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that translate a question to a SQL query given a database."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
                top_p=1.0,
                stop=stop,
            )
            break
        except Exception as e:
            print(e, "Retry.")
            time.sleep(10)
            continue
    for i in range(len(response["choices"])):
        x = response["choices"][i]["message"]["content"].replace('\n', ' ').replace('    ', ' ').replace('\t', ' ').replace('\r', ' ')
        for s in stop:
            if s in x:
                x = x[:x.index(s)]
        response["choices"][i]["text"] = ' ' + x
    return response


def call_codex(model, prompt, max_tokens=200, stop=[";", "Question", 'Answer', '/*'], num_return=1, temperature=0, top_p=1):
    api_model = "code-davinci-002"
    while (True):
        try:
            response = openai.Completion.create(
                model=api_model,
                prompt=prompt,
                n=num_return,
                best_of=num_return,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop,
                logprobs=5
            )
            break
        except Exception as e:
            print(e, "Retry.")
            time.sleep(10)
            continue
    for i in range(len(response["choices"])):
        response["choices"][i]["text"] = response["choices"][i]["text"].replace('\n', ' ').replace('    ', ' ').replace('\t', ' ').replace('\r', ' ')
    return response


def sql_generation(model, prompt):
    if model == "codex":
        return call_codex(model, prompt)
    if model in ["chatgpt", "chatgpt16k", "gpt4"]:
        return call_chatgpt(model, prompt)

def text_to_sql_direct(model, questions, db_prompt,save_prompt_only=False):
    if model == "gpt3.5":
        api_name = "text-davinci-003"
    elif model == "codex":
        api_name = "code-davinci-002"
    elif model == "chatgpt":
        api_name = "gpt-3.5-turbo"
    elif model == "chatgpt16k":
        api_name = "gpt-3.5-turbo-16k-0613"
    elif model == "gpt4":
        api_name = "gpt-4-0613"
    elif model in ["llama34instruct", "llama7", "llama13"]:
        api_name = None
    else:
        raise NotImplementedError

    predictions = []
    prompts = []
    prompts_len = []
    stop = [";", "Question", 'Answer', '/*']
    for q in questions:
        prompt = db_prompt + f"Question: {q['question']}\n" + "select"
        prompt_len = get_prompt_length(model, prompt)
        prompts.append(prompt)
        prompts_len.append(prompt_len)

    if save_prompt_only:
        return prompts, []

    responses = []
    for q, prompt, prompt_len in zip(questions, prompts, prompts_len):
        response = sql_generation(model, prompt)
        responses.append(response)
        sql = "select" + response["choices"][0]["text"]
        # print(prompt)
        print(q["question"])
        print(sql)
        print(prompt_len)

    for q, response, prompt_len in zip(questions, responses, prompts_len):
        sql = "select" + response["choices"][0]["text"]
        predictions.append({
            "db_id": q["db_id"],
            "question": q["question"],
            "gold_sql": q['gold']['query_normalized'],
            "predicted_sql": sql,
            "prompt_len": prompt_len,
        })
    return prompts, predictions



def text_to_sql_few_shot_indomain(model, questions, indomain_schema, indomain_demo_examples_per_question, demo_sql_format="normalized", save_prompt_only=False):
    print("=" * 10 + "start" + "=" * 10)
    few_shot_in_prompts = []
    predictions = []
    prompts = []
    prompts_len=[]
    
    for q, indomain_few_shot_examples in zip(questions, indomain_demo_examples_per_question):
        prompt = indomain_schema
        indomain_demonstration = []
        for example in indomain_few_shot_examples:
            prompt += f"Question: {example['question']}\n"
            query = format_query(example, demo_sql_format)
            prompt += query + '\n'
            indomain_demonstration.append([example["question"], query])

        few_shot_in_prompts.append([q["question"], q["query"], indomain_demonstration])
        prompt += f"Question: {q['question']}\n" + "select"
        prompt_len = get_prompt_length(model, prompt) 
        prompts.append(prompt)
        prompts_len.append(prompt_len)

    if save_prompt_only:
        return [], prompts, []
    
    for q, prompt, prompt_len in zip(questions, prompts, prompts_len):
        response = sql_generation(model=model, prompt=prompt)
        print(response)

        sql = "select" + response["choices"][0]["text"]
        print(q["question"])
        print(sql)
        predictions.append({
            "db_id": q["db_id"],
            "question": q["question"],
            "gold_sql": q['gold']['query_normalized'],
            "predicted_sql": sql,
            "prompt_len": prompt_len,
        })
    return few_shot_in_prompts, prompts, predictions



def create_outdomain_prompt(outdomain_schemas, outdomain_demo_examples,  demo_sql_format="normalized"):
    prompt = ""
    outdomain_demostration = []
    for schema, examples in zip(outdomain_schemas, outdomain_demo_examples):
        prompt += DB_SEP
        prompt += schema
        outdomain_demostration.append([])
        for example in examples:
            prompt += f"Question: {example['question']}\n"
            query = format_query(example, demo_sql_format)
            prompt += query + '\n'
            outdomain_demostration[-1].append([example["question"], query])
        prompt += '\n'
    return prompt, outdomain_demostration


def text_to_sql_few_shot_outdomain(model, questions, outdomain_schemas_per_question, indomain_schema, outdomain_demo_examples_per_question,
                                   demo_sql_format="normalized",save_prompt_only=False):

    few_shot_in_prompts = []
    print("=" * 10 + "start" + "=" * 10)
    predictions = []
    prompts = []
    prompts_len = []
    for q, outdomain_schemas, outdomain_demo_examples in zip(questions, outdomain_schemas_per_question, outdomain_demo_examples_per_question):
        prompt, outdomain_demostration = create_outdomain_prompt(outdomain_schemas, outdomain_demo_examples, demo_sql_format=demo_sql_format)
        prompt += DB_SEP
        prompt += indomain_schema
        few_shot_in_prompts.append([q["question"], q["query"], outdomain_demostration])
        prompt += f"Question: {q['question']}\n" + "select"
        prompt, prompt_len = cut_prompt_with_max_tokens(model, prompt, MAX_GEN_TOKENS)
        prompts.append(prompt)
        prompts_len.append(prompt_len)
    if save_prompt_only:
        return [], prompts, []
    for q, prompt, prompt_len in zip(questions, prompts, prompts_len):
        response = sql_generation(model=model, prompt=prompt)
    
        sql = "select" + response["choices"][0]["text"]

        # print(prompt)
        # print(q["question"])
        # print(sql)
        # print(prompt_len)
        predictions.append({
            "db_id": q["db_id"],
            "question": q["question"],
            "gold_sql": q['gold']['query_normalized'],
            "predicted_sql": sql,
            "prompt_len": prompt_len,
        })
        break
    return prompts, few_shot_in_prompts, predictions


def text_to_sql_few_shot_inoutdomain(model, questions, outdomain_schemas_per_question, indomain_schema, outdomain_demo_examples_per_question,
                                     indomain_demo_examples_per_question, demo_sql_format="normalized",save_prompt_only=False):
    
    few_shot_in_prompts = []
    print("=" * 10 + "start" + "=" * 10)
    predictions = []
    prompts = []
    prompts_len = []
    for q, outdomain_schemas, outdomain_demo_examples, indomain_few_shot_examples in  zip(questions, outdomain_schemas_per_question, outdomain_demo_examples_per_question, indomain_demo_examples_per_question):
        prompt, outdomain_demostration = create_outdomain_prompt(outdomain_schemas, outdomain_demo_examples, demo_sql_format=demo_sql_format)
        prompt += DB_SEP
        prompt += indomain_schema
        indomain_demonstration = []
        for example in indomain_few_shot_examples:
            prompt += f"Question: {example['question']}\n"
            query = format_query(example, demo_sql_format)
            prompt += query + '\n'
            indomain_demonstration.append([example["question"], query])

        few_shot_in_prompts.append([q["question"], q["query"], outdomain_demostration, indomain_demonstration])
        
        prompt += f"Question: {q['question']}\n" + "select"

        prompt, prompt_len = cut_prompt_with_max_tokens(model, prompt, MAX_GEN_TOKENS)
        prompts.append(prompt)
        prompts_len.append(prompt_len)
    if save_prompt_only:
        return [], prompts, []
    for q, prompt,prompt_len in zip(questions, prompts,prompts_len):

        response = sql_generation(model=model, prompt=prompt)
        sql = "select" + response["choices"][0]["text"].replace('\n', ' ').replace('\t', ' ').replace('    ', ' ')
        print(prompt)
        print(q["question"])
        print(sql)
        print(prompt_len)
        print()
        predictions.append({
            "db_id": q["db_id"],
            "question": q["question"],
            "gold_sql": q['gold']['query_normalized'],
            "predicted_sql": sql,
            "prompt_len": prompt_len,
        })
        break

    return prompts, few_shot_in_prompts, predictions
