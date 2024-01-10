import openai
# import tiktoken
import json
import copy as cp
import argparse

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def generate_readme(data, openai_key):
    # tokenizer = tiktoken.get_encoding("cl100k_base")
    openai.api_key = openai_key
    files = [x['contents'] for x in data]

    ## send each file with a summarizing prompt
    responses=[]
    for f in files:
        prompt1 = f'''
        You are a expert developer , you want to give back a summary of the most valuable information
    This information will be later needed to write a README . It also needs to mention the main dependencies
    and installation precedures
    The following codefile is below delimited by triple backticks
    ```{f}```
    '''
        response = get_completion(prompt1)
        print("done!")
        responses.append(cp.deepcopy(response))

    print("responses are done")
    responses_all =''
    for resp in responses:
        responses_all+=resp

    prompt_final = f'''
    You are a expert developer , you need to synthesize a README from the given summaries .
    Each summary is for one file in the project
    We want an output with markdown.

    The following summaries are below   delimited by triple backticks.
    ```{responses_all}```
    '''
    response = get_completion(prompt_final)
    return response





