import yaml
import json
import pandas as pd
import openai
from requests.exceptions import RetryError
from tenacity import retry, stop_after_attempt, wait_exponential
from enum import Enum
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

class Operation(Enum):
    CATEGORIZE = "categorize"
    NORMALIZE = "normalize"
    ENRICH = "enrich"

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def enrich_data(context, responses, subcategories, model):
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": f"""You are a sophisticated assistant trained to enrich data according to the following rules:
                - {context}
                - Add additional relevant information to each record.
                - Enhance the existing data with external sources or APIs.
                - Validate and correct data inconsistencies.
                - Perform feature engineering or extraction.
                - Perform any other data enrichment tasks based on the provided context and requirements."""
            },
            {
                "role": "assistant",
                "content": f"""
                Given the rules I'm trained with, please help me enrich the following records.
                ###
                {responses}
                ###
                Desired Format: JSON with the all enrichment data under processed key. Example: {{
                    "records": [
                        {{ 
                            "record_id":"None",
                            "processed": "None"
                        }}
                    ]
                }}
                """,
            },
        ],
    }

    try:
        response = openai.ChatCompletion.create(**payload)
        content = response['choices'][0]['message']['content']
        responses = json.loads(content)['records']

        print("Response Content:", content)  # Debug statement
        
        return responses
    except openai.error.ServiceUnavailableError as e:
        print("Service Unavailable Error. Retrying...")
        raise RetryError(attempt=e.last_attempt)  # Raising RetryError to trigger retry with exponential backoff
    except Exception as e:
        print("Error in JSON Parse", e)
        return ["Other" for _ in responses]

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def normalize_response(context, responses, subcategories, model):
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": f"""You are a sophisticated assistant trained to normalize responses according to the following rules:
                - {context}
                - Convert all text to lowercase.
                - Remove any extra white spaces.
                - Standardize dates to the 'YYYY-MM-DD' format.
                - Remove or replace special characters, unless part of a URL or email address.
                - Replace missing or null values with 'unknown'.
                - Expand common abbreviations.
                - Convert all currencies to USD and standardize number formats (ie: $1.00)
                - Standardize text encoding to UTF-8.
                - Extract domain name from URLs or email addresses.
                - Correct commonly misspelled words.
                For any response, ensure it follows the correct format, is free from typographical errors, and has consistent capitalization and punctuation."""
            },
            {
                "role": "assistant",
                "content": f"""
                Given the rules I'm trained with, please help me normalize the following records.
                ###
                {responses}
                ###
                Desired Format: JSON with the key normalized. Example: {{
                    "records": [
                        {{ 
                            "record_id":"None"
                            "processed":"None"
                        }}
                    ]
                }}
                """,
            },
        ],
    }

    try:
        response = openai.ChatCompletion.create(**payload)
        content = response['choices'][0]['message']['content']
        print("Response Content:", content)  # Debug statement
        responses = json.loads(content)['records']
        return responses
    except openai.error.ServiceUnavailableError as e:
        print("Service Unavailable Error. Retrying...")
        raise RetryError(attempt=e.last_attempt)  # Raising RetryError to trigger retry with exponential backoff
    except Exception as e:
        print("Error in JSON Parse", e)
        return ["Other" for _ in responses]

def load_tasks(filename):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded['tasks']

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def categorize_response(context, responses, subcategories, model):

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "You are a survey response classifier taking user response "
                           "and converting it to the most likely subcategories provided",
            },
            {
                "role": "assistant",
                "content": f"""
                Question: {context}
                Here are the possible subcategories: {', '.join(subcategories)}. 
                Please categorize the following responses: 
                ###
                {responses}
                ###
                Desired Format: JSON with the key processed. Make sure 
                Example: {{
                    "responses": [
                        {{ 
                            "record_id":"None"
                            "processed":"None"
                        }}
                    ]
                }}
                """,
            },
        ],
    }

    try:
        response = openai.ChatCompletion.create(**payload)
        content = response['choices'][0]['message']['content']
        print("Response Content:", content)  # Debug statement
        responses = json.loads(content)['responses']
        return responses
    except openai.error.ServiceUnavailableError as e:
        print("Service Unavailable Error. Retrying...")
        raise RetryError(attempt=e.last_attempt)  # Raising RetryError to trigger retry with exponential backoff
    except Exception as e:
        print("Error in JSON Parse", e)
        return ["Other" for _ in responses]
    
operation_to_function_map = {
    Operation.CATEGORIZE: categorize_response,
    Operation.NORMALIZE: normalize_response,
    Operation.ENRICH: enrich_data,
}

def process_data(tasks):
    for task in tasks:
        input_file = task.get("input_file")
        batch_size = task.get("batch_size", 50)
        survey_data = pd.read_csv(input_file, encoding='ISO-8859-1')
        output_file = task.get("output_file", "output_file.csv")
        model = task.get("model", "gpt-3.5-turbo")
        num_rows = len(survey_data)
        for i in range(0, num_rows, batch_size):
            batch_data = survey_data.iloc[i:min(i+batch_size, num_rows)].copy()

            context = task.get("context", "")
            operation_str = task.get("operation")
            id_column = task.get("id_column")

            try:
                operation = Operation[operation_str.upper()]
            except KeyError:
                print(f'Operation {operation_str} not found. Skipping...')
                continue

            process_func = operation_to_function_map[operation]
            subcategories = [subtask.get("name") for subtask in task.get("subcategories", [])]
            response_key = task.get("data_column")

            batch_responses = batch_data[response_key].tolist()
            batch_ids = batch_data[id_column].tolist()

            if batch_responses and batch_ids:
                responses_stringified = "\n".join([f"{id_},{resp}" for id_, resp in zip(batch_ids, batch_responses)])
                processed_responses = process_func(context, responses_stringified, subcategories, model)
                processed_subcategories = [resp['processed'] for resp in processed_responses]
                
                batch_data.loc[:, response_key] = processed_subcategories

            new_data = pd.DataFrame(batch_data)
            mode = 'a' if i > 0 else 'w'  # write mode, use 'a' for append if it's not the first batch
            new_data.to_csv(output_file, mode=mode, header=(i == 0), index=False)


if __name__ == "__main__":
    # load tasks from yaml file
    tasks = load_tasks("config.yaml")

    # process the data
    process_data(tasks)
