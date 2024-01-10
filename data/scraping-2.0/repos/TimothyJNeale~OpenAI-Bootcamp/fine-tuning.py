# Fine tune a model
####################################### IMPORTS ###############################################
import openai
import logging
import os

import pandas as pd
import tiktoken
import json

from dotenv import load_dotenv

##################################### CONSTANTS ###############################################

DATA_DIRECTORY ='data'
DATA_FILE = 'python_qa.csv'
DATASET_SIZE = 500
TRAINING_COST_PER_TOKEN = 0.0004
NUM_EPOCHS = 4

# load environment variables from .env file
load_dotenv()

############################################ Data #############################################


################################## HELPER FUCTIONS #############################################

# Use chat completion
def get_chat_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message["content"]

# Standard completion
def get_completion(prompt, model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=300, stop="\"\"\""):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop)

    return response.choices[0].text

# Get the number of tokens in a prompt
def get_num_tokens_from_string(string, encoding_name="gpt2"):
    tokenizer = tiktoken.get_encoding(encoding_name)
    tokens = tokenizer.encode(string)

    return len(tokens)


####################################### LOGGING ################################################

logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.DEBUG) # Supress debugging output from modules imported
#logging.disable(logging.CRITICAL) # Uncomment to disable all logging


####################################### START #################################################
logging.info('Start of program')

# Get the current DATA directory
home = os.getcwd()
data_dir = os.path.join(home, DATA_DIRECTORY)
logging.info(data_dir)

os.chdir(data_dir)

# Authenticate with OpenAI                             
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

####################################### MAIN ##################################################
# Load the data
input_file = os.path.join(data_dir, DATA_FILE)
logging.info(input_file)
qa_df = pd.read_csv(input_file)
logging.info(qa_df.head())

questions, answers = qa_df['Body'], qa_df['Answer']
logging.info(questions.head())
logging.info(answers.head())

# Format the data for fine-tuning
qa_openai_format = [ {"prompt": q, "completion": a} for q, a in zip(questions, answers)]
logging.info(qa_openai_format[10])

# # Test the prompt
# response = get_completion(
#     model = "babbage-002",
#     prompt = qa_openai_format[10]["prompt"])

# logging.info(response)

# # Test the prompt
# response = get_completion(prompt = qa_openai_format[10]["prompt"])

# logging.info(response)

# Write the processes q qnd a fike to json file
with open("example_training_data.json", "w") as f:
    for entry in qa_openai_format[:DATASET_SIZE]:
        f.write(json.dumps(entry) + "\n")

# Estimate the costs of the fine-tuning using tiktoken library
token_counter = 0
for entry in qa_openai_format[:DATASET_SIZE]:
    for prompt, completion in entry.items():
        token_counter += get_num_tokens_from_string(prompt)
        token_counter += get_num_tokens_from_string(completion)

logging.info(f"Total number of tokens: {token_counter}")
costs = token_counter * TRAINING_COST_PER_TOKEN * NUM_EPOCHS / 1000
logging.info(f"Estimated cost: ${costs}")


################################### FINE TUNE VIA COMMAND LINE #########################################