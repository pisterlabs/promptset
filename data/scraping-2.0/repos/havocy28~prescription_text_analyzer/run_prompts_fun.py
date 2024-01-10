'''This script runs prompts in the stored dataframe'''

import asyncio
import openai
import pandas as pd
import pickle
import os
from time import sleep
from tqdm import tqdm
import time
import concurrent.futures
import random
import json
from datetime import datetime

import os

# # Verify environment variables
openai_organization = os.getenv('OPENAI_ORGANIZATION')
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_organization:
    raise Exception("Missing value for OPENAI_ORGANIZATION environment variable.")

if not openai_api_key:
    raise Exception("Missing value for OPENAI_API_KEY environment variable.")

# Set OpenAI credentials
openai.organization = openai_organization
openai.api_key = openai_api_key



class CFG:
    # parquet_file = './outputs/prompts/test_prompts_v5.parquet'
    # parquet_file = './outputs/prompts/test_ingredient_prompts_v1.parquet'
    # parquet_file = './outputs/prompts/test_similar_prompts_v1.parquet'
    ingredients_file = './ingredient_list.csv'
    input_file = './synthetic_rx.txt'
    # labels_file = './deid_rx_labels_v4.parquet'
    # prompt_columns = ['similar_all_prompts', 'similar_all_prompt_instruct', 'zesh_all_prompts', 'zesh_len_prompts', 'len_prompts', 'all_prompts', 'all_prompts_instruct']  # now a list of columns
    prompt_columns = ['prompts'],
    #    'similar_all_prompts', 'similar_all_prompt_instruct',
    #    'zesh_all_prompts', 'zesh_len_prompts', 'zesh_dose_prompts',
    #    'fun_prompts', 'dose_prompts', 'med_unit_prompts',
    #    'zesh_med_unit_prompts', 'weight_prompts', 'zesh_weight_prompts',
    #    'dose_size_prompts', 'zesh_dose_size_prompts',
    #    'zesh_units_dispensed_prompts', 'units_dispensed_prompts'] 
    # prompt_columns = ['similar_ingredient_prompts', 'ingredient_prompts', 'zesh_ingredient_prompts']
    # prompt_columns = ['similar_weight_prompts']
    # prompt_columns = ['zesh_all_prompts', 'zesh_len_prompts']  # now a list of columns
    # prompt_columns = ['fun_prompts'] 
    batch_size = 5
    cache_file = 'cache_{}.pkl'
    clear_cache = True
    # output_file = './outputs/dose_predictions.parquet'
    retries = 10
    min_sleep = 40  # minimum sleep time in seconds
    max_sleep = 60  # maximum sleep time in seconds
    model_name = 'gpt-3.5-turbo-0613'
    # model_name = 'gpt-4-0314'
    # model_name = 'gpt-4'
    # output_suffix = '_GPT4'
    # output_suffix = '_GPT3_v1'
    function_test = True
    temperature = 0
    # token_limit = 50000000000  # set your desired token limit here
    # request_limit = 3000   # set your desired request limit here
    timeout = 20
    SEEDS = [1]

rx_ingredients = pd.read_csv(CFG.ingredients_file)['Ingredient'].unique().tolist()
print(rx_ingredients)
CFG.functions = [
    {
        "name": "dosing_information",
        "description": "information about dosing in the prescription label",
        "parameters": {
            "type": "object",
            "properties": {
                "medication_unit_size": {
                    "type": "number",
                    "description": "Unit size of medication in mg"
                },
                "dose_unit_size": {
                    "type": "number",
                    "description": "Units of medication given per dose"
                },
                "patient_weight": {
                    "type": "number",
                    "description": "Weight of patient in kg"
                },
                "dosing_freq": {
                    "type": "number",
                    "description" : "number of times per day the medication is given"
                },
                "units_dispensed": {
                    "type" : "number",
                    "description" : "number of medication units dispensed"
                },
                "ingredient": {
                "type": "string", 
                "enum": rx_ingredients,
                "description": "Referencing the trade name, identify each active ingredient that forms the medication. For combination drugs, ensure to list all components."
            },
            },
            "required": ["medication_unit_size", "dose_unit_size", "patient_weight", "dosing_freq", "units_dispensed", "ingredient"]
        }
    },
    
]



async def create_chat_completion(prompt):
    retries = CFG.retries
    sleep = CFG.min_sleep  # initial sleep duration

    
    for attempt in range(retries):
        try:
            chat_completion_resp = await openai.ChatCompletion.acreate(
                model=CFG.model_name, 
                messages=[{"role": "user", "content": prompt}],
                timeout=CFG.timeout,
                temperature=CFG.temperature
            )
            response = chat_completion_resp['choices'][0]['message']['content']
            
            # If response is not empty, return it even if an error is caught
            if response:
                await asyncio.sleep(0.001)  # Sleep for 0.001 seconds after every call
                return response
            else:
                print(f"Empty response on attempt {attempt + 1}. Retrying.")

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:  # if not on final attempt
                sleep = min(2 * sleep + random.random(), CFG.max_sleep)  # exponential backoff with random jitter
                print(f"Sleeping for {sleep} seconds before next attempt.")
                await asyncio.sleep(sleep)
            else:
                return None
 # return None after too many failed attempts
 
 
def get_dose(medication_unit_size, dose_unit_size, patient_weight, dosing_freq, units_dispensed, ingredient):
    """Get the dose and length of administration of the medication given"""
    dose_info = [
        {
        "dose": (medication_unit_size * dose_unit_size) / patient_weight
        },
        {
        "length_admin": units_dispensed / (dose_unit_size * dosing_freq)
        },
        {
        "dose_freq": dosing_freq
        },
        {
        "units_dispensed": units_dispensed
        },
        {
        "medication_unit_size": medication_unit_size
        },
        {
        "dose_unit_size": dose_unit_size
        },
        {
        "patient_weight": patient_weight,
        },
        {
        "ingredient" : ingredient
        }
        
    ]
    return json.dumps(dose_info)

async def create_fun_chat_completion(prompt, functions=CFG.functions):
    retries = CFG.retries
    sleep = CFG.min_sleep  # initial sleep duration
    PREFIX = 'What is the dosing information for the medication given **\\ Instance to Label: '

        
    for attempt in range(retries):
        try:
            print(f'Prompt: {PREFIX + prompt}')
            prompt = PREFIX + prompt
            response = await openai.ChatCompletion.acreate(
                model=CFG.model_name, 
                messages=[{"role": "user", "content": prompt}],
                functions=functions,
                function_call="auto",
                timeout=CFG.timeout
            )
            # print(chat_completion_resp)
            # response = chat_completion_resp['choices'][0]['message']['content']
            
            # If response is not empty, return it even if an error is caught
            if response:
                await asyncio.sleep(0.001)  # Sleep for 0.001 seconds after every call
                response_message = response["choices"][0]["message"]
                if response_message.get("function_call"):
                    available_functions = {
                        "dosing_information": get_dose,
            #             "length_of_administration_information": get_length_of_administration,
                    
                    }
                    
                    function_name = response_message["function_call"]["name"]
                    function_to_call = available_functions[function_name]
                    try:
                        function_args = json.loads(response_message["function_call"]["arguments"])
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error on attempt {attempt + 1}: {e}")
                        CFG.max_sleep = 0.5
                        # continue  # This skips the rest of this iteration and starts the next one

                    
                    # function_args = json.loads(response_message["function_call"]["arguments"])
                    function_response = function_to_call(
                        medication_unit_size=function_args.get("medication_unit_size"),
                        dose_unit_size=function_args.get("dose_unit_size"),
                        patient_weight=function_args.get("patient_weight"),
                        dosing_freq=function_args.get("dosing_freq"),
                        units_dispensed=function_args.get("units_dispensed"),
                        ingredient=function_args.get("ingredient"),
                        
                    )
                return function_response
            else:
                CFG.max_sleep = 0.5
                print(f"Empty response on attempt {attempt + 1}. Retrying.")

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            # print(f'Prompt: {prompt}')
            # print(f'Response: {response}')
            if attempt < retries - 1:  # if not on final attempt
                sleep = min(2 * sleep + random.random(), CFG.max_sleep)  # exponential backoff with random jitter
                print(f"Sleeping for {sleep} seconds before next attempt.")
                await asyncio.sleep(sleep)
            else:
                return None 



async def return_cached(response):
    return response

async def process_prompts(df):
    responses = {}
    for j, col in enumerate(CFG.prompt_columns, start=1):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Processing column {j}/{len(CFG.prompt_columns)}: {col}")

        # Retry logic for the entire column
        retries = CFG.retries  # Maximum number of retries for a column
        while retries:
            try:
                # Your existing logic here
                if os.path.exists(CFG.cache_file.format(col)) and not CFG.clear_cache:
                    with open(CFG.cache_file.format(col), 'rb') as f:
                        col_responses = pickle.load(f)
                else:
                    col_responses = [None] * len(df)

                progress_bar = tqdm(total=len(df), ncols=70)  # setting up progress bar
                for i in range(0, len(df), CFG.batch_size):
                    batch = df.iloc[i:i + CFG.batch_size]
                    tasks = [
                        asyncio.wait_for(create_fun_chat_completion(prompt), timeout=1200) if response is None and CFG.function_test else
                        asyncio.wait_for(create_chat_completion(prompt), timeout=1200) if response is None else
                        return_cached(response)
                        for prompt, response in zip(batch[col], col_responses[i:i + CFG.batch_size])
                    ]
                    batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Handle exceptions and assign valid responses
                    for idx, resp in enumerate(batch_responses):
                        if isinstance(resp, Exception):
                            print(f"Task failed due to {resp}")
                        else:
                            col_responses[i + idx] = resp

                    with open(CFG.cache_file.format(col), 'wb') as f:
                        pickle.dump(col_responses, f)

                    progress_bar.update(len(batch))

                progress_bar.close()
                responses[col] = col_responses
                break  # Exit the retry loop for this column
                
            except asyncio.TimeoutError:
                retries -= 1
                print(f"Column {col} timed out. Retries remaining: {retries}")

    return responses


# '''multiple seed process prompts section'''
# # seed_list = [5, 10, 15]  # Add other seed numbers as required
# dfs = []  # List to hold DataFrames for each seed

# # Initialize event loop
# loop = asyncio.get_event_loop()

# for seed in CFG.SEEDS:
#     CFG.seed = seed  # Dynamically set the seed
#     intermediate_file = f"{CFG.output_file}_seed_{seed}.parquet"

#     # Check if this seed's file already exists
#     if os.path.exists(intermediate_file):
#         print(f"File for seed {seed} already exists. Reading file.")
#         df = pd.read_parquet(intermediate_file)
#         dfs.append(df)
#         continue

#     print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Running for seed {seed}...")

try:
    # Load DataFrame
    # Initialize an empty list to hold the lines from the file
    lines_list = []

    # Read the file line by line
    with open(CFG.input_file, "r") as file:
        for line in file:
            lines_list.append(line.strip())
    df = pd.DataFrame(lines_list, columns=CFG.prompt_columns[0])
    print(df.columns)
    print(df.head())
    # df = pd.read_parquet(CFG.parquet_file)[:200]
    # assert all(col in df.columns for col in CFG.prompt_columns), "Not all prompt columns are present in the DataFrame."
    # df.reset_index(inplace=True, drop=True)  # Reset index and drop the old index column
    loop = asyncio.get_event_loop()
    # Run process_prompts
    print(f"loading files - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    responses = loop.run_until_complete(process_prompts(df))
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Completed run")
    


    # # Add responses to DataFrame
    # for col in CFG.prompt_columns:
    #     new_col_name = f"{col}_{CFG.output_suffix}"  # Append seed number to new column name
    #     df[new_col_name] = responses[col]

    # # Save intermediate DataFrame with seed information
    # df['seed'] = seed
    # df.to_parquet(intermediate_file, index=False)  # Save without index

    # # Append DataFrame to dfs list
    # dfs.append(df)

except Exception as e:
    print(f"An error occurred for processing file: {e}")


print("writing outputs")
# Save the final concatenated DataFrame
responses.to_parquet(CFG.output_file, index=False)  # Save without index


print("Done!")