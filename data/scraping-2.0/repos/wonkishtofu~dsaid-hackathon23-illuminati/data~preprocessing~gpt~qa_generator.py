"""
HOW TO USE

dependency: requirements.txt, all scraped raw data files in ../../raw/

outputs: raw_data_summaries_sample.csv, raw_data_qna_sample.csv
"""

import backoff
import csv
from dotenv import load_dotenv, find_dotenv
import json
import openai
import os
import pandas as pd
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff (to overcome rate limit)
import time
from transformers import pipeline


""" LOAD OPENAI_API_KET FROM ENV """
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

current_path = os.getcwd()
raw_path = os.chdir('../../raw/')

""" READ SCRAPED RAW DATA FILES (CSV/JSON) """

raw_data = {}
for filename in os.listdir(raw_path):
    if filename == "nccs_website_transcript.csv":
        name, file_extension = os.path.splitext(filename)
        # read CSV raw data files
        if '.csv' in file_extension:
            input_df = pd.read_csv(name + file_extension)
            
            if 'pdf' in name: # concatenate headers & subheaders for PDF title
                for ii, text in enumerate(input_df['Text']):
                    title = ""
                    for cc in input_df.columns:
                        if cc == 'Text': break
                        else: title += input_df[cc][ii] + " | "
                    raw_data[title] = input_df['Text'][ii]
                    
                print("Successfully read file: {}".format(name + file_extension))
            else:
                # extract (title, content) pair and save in raw_data
                for ii, title in enumerate(input_df['Title']):
                    raw_data[title] = input_df['Content'][ii]
                    
                print("Successfully read file: {}".format(name + file_extension))
        
        # read JSON raw data files
        elif '.json' in file_extension:
            with open(name + file_extension, 'r') as f:
                data = json.load(f)
            # extract (key, value) pair and save in raw_data
            for key, value in data.items():
                raw_data[key] = value
                
            print("Successfully read file: {}".format(name + file_extension))
            
        # unread files
        else:
            print("Did not read file: {}".format(name + file_extension))

# change working directory to save GPT outputs
processed_path = os.chdir('../processed/')

""" IMPLEMENT EXPONENTIAL BACKOFF """
# Exponential backoff decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
):
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                # Sleep for the delay
                time.sleep(delay)
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
                
    return wrapper
    
@retry_with_exponential_backoff
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# Helper function to get returns from GPT
def get_message_completion(messages,
                     model = "gpt-3.5-turbo",
                     temperature = 0):
    response = completion_with_backoff(
        model = model,
        messages = messages,
        temperature = temperature, # this is the degree of randomness of the model's output
    )
    usage = response.usage.total_tokens
    
    return response.choices[0].message.content, usage

# Function to get response and usage, retrying based on errors encountered
def chat(prompt):
    
    try:
        response, usage = get_message_completion(prompt)
        print("Response Successfully Generated!")
        print("Usage Details: {} \n".format(usage))
        return response
        
    except openai.error.RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return chat(prompt)
        
    except openai.error.APIError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"API error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return chat(prompt)
        
    except OSError as e:
        retry_time = 30  # Adjust the retry time as needed
        print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return chat(prompt)
        
extracts = {}
qna = {}
num_qna = 10 # Set number of Q&As

""" LOOP THROUGH SCRAPED RAW DATA FILES TO
(A) SUMMARISE
(B) GENERATE Q&A PAIRINGS """


for title, content in raw_data.items():
    summarise_prompt = [{'role':'system',
         'content':f"Please refer to the content provided: \n {content.strip()} \n\n"},
        {'role':'user',
         'content':f"Summarise the most relevant lines. Make sure to retain key statistics or figures."},]
    extracts[title] = chat(summarise_prompt)
    
print(len(extracts))
extracts_df = pd.DataFrame(
    [(k, val) for k, val in extracts.items()],
    columns = ['Title', 'Summary']
)
extracts_df.to_csv("raw_data_summaries_sample.csv")

    
for title, summary in extracts.items():
#for title, content in raw_data.items():
    QA_prompt =  [{'role':'system',
         'content':f"""Please refer to the summary provided: {summary.strip()}"""},
        {'role':'user',
         'content':f"""Create a JSON of {num_qna} pairs of questions and answers based on this content. \
         The key value pairs should be the question and answer. The resultant dictionary should be of the following format: ("1": ("question": "INSERT_QUESTION_HERE", "answer", "INSERT_ANSWER_HERE"), "2": (), ...)"""},]
    qna_dict = dict(json.loads(chat(QA_prompt)))
    
    # Create list of tuples based on Q&A pairings and save to dictionary
    qna[title] = [(qna_dict[k]['question'], qna_dict[k]['answer']) for k in qna_dict.keys()]
    
    print(qna[title])
    
print(len(qna))

# Create df and save to csv
qna_df = pd.DataFrame(qna)
qna_melt = pd.melt(qna_df, value_vars = [k for k in qna.keys()], var_name = "Title")
qna_melt['Questions'], qna_melt['answers'] = zip(*qna_melt['value'])
qna_melt = qna_melt.drop('value', axis = 1)

qna_melt.to_csv("raw_data_qna_sample.csv")
