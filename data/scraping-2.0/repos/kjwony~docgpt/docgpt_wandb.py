# Description: This script gets the input information from tsv file and generates the output from GPT model
#             and stores the output in W&B table.

import os
import time
import datetime
import openai
import wandb
import pandas as pd

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,# for exponential backoff
)

# Set this to `azure` or do not set this for OpenAI API 
os.environ["OPENAI_API_TYPE"] = "azure"
openai.api_type = os.environ["OPENAI_API_TYPE"]

# set openai API key
os.environ['OPENAI_API_KEY'] = "your key"
openai.api_key  = os.environ['OPENAI_API_KEY']

# set openai API version
openai.api_version = "your version"

# set openai API base
openai.api_base = "your base"

PROJECT = "docgpt_wandb"
MODEL_NAME = "your model name"
TASK_TYPE = "my task"

# Login to W&B to see gpt output
wandb.login()
run = wandb.init(project=PROJECT, job_type="generation", group=f"GROUP:{TASK_TYPE}", name="my run")

# Define function to retry on failure
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))

def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def generate_and_print(system_prompt, user_prompt, table, n=1):
    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    start_time = time.time()
    responses = completion_with_backoff(
        engine=MODEL_NAME,
        messages=messages,
        n = n,
        )
    elapsed_time = time.time() - start_time
    for response in responses.choices:
        generation = response.message.content
        print(generation)
    table.add_data(system_prompt,
                user_prompt,
                [response.message.content for response in responses.choices],
                elapsed_time,
                datetime.datetime.fromtimestamp(responses.created),
                responses.model,
                responses.usage.prompt_tokens,
                responses.usage.completion_tokens,
                responses.usage.total_tokens
                )

# Define W&B Table to store generations
columns = ["system_prompt", "user_prompt", "generations", "elapsed_time", "timestamp",\
            "model", "prompt_tokens", "completion_tokens", "total_tokens"]
table = wandb.Table(columns=columns)

# Get data from doc.tsv
df = pd.read_csv("doc.tsv", sep="\t")
for index, row in df.iterrows():
    system_prompt = row["system_prompt"]
    context1 = row["context1"]
    context2= row["context2"]
    context3 = row["context3"]
    question = row["question"]
    user_prompt = """문서 1: {context1}\n문서 2: {context2}\n문서 3: {context3}\n질문: {question}\n한국어 답변:""".format(context1=context1, context2=context2, context3=context3, question=question)
    generate_and_print(system_prompt, user_prompt, table)

wandb.log({"의료지식 GPT ": table})
run.finish()