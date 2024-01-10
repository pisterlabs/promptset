import os
with open("./openai-api-.txt", "r") as file: 
    api = file.read()
os.environ["OPENAI_API_KEY"] = api
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import time
import pandas as pd
import re
import json
import random
import numpy as np
from tqdm import tqdm


def get_gpt_response(question, system_prompt="", temperature=0.7, 
                     model="gpt-3.5-turbo"):

  completion = None
  while completion is None:
  
    completion = client.chat.completions.create(model="gpt-3.5-turbo",
    temperature=temperature,
    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": f"```{question}```"}
    ])
    return completion.choices[0].message.content
  


def get_gpt4_response(question, system_prompt="", temperature=0.7):

  try:
    completion = client.chat.completions.create(model="gpt-4",
    temperature=temperature,
    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": f"{question}"}
    ])
  except:
    print("\nFirst request failed... Trying in 3 seconds \n")
    time.sleep(3)
    completion = client.chat.completions.create(model="gpt-4",
    temperature=temperature,
    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": f"{question}"}
    ])

  return completion.choices[0].message['content']



def get_prompt(sentence, a, b, c, d):
    system_prompt = f"""
    You are given a science multiple choice question and their options.
    Your job is to correctly answer the question. First reason step by step and only then give me the final answer as "a", "b", "c" or "d". Only include the letter in your answer. Do not include the option text.

    Your answer should be in this format: {{"Answer": "final answer"}}. The question is given below within triple ticks ```:\n
    
    """
    options = f"(a) {a} \n(b) {b} \n(c) {c}\n(d) {d}"
    question = f"```Question: {sentence} \n Options: {options}```"

    final_prompt = question + "Let's think step by step and answer"

    return final_prompt

def rename_cols(df):
    col_names = list(df.columns)
    df = df.rename(columns={col_names[0]: "Question",
    col_names[1]: "a", col_names[2]: "b", col_names[3]: "c",
    col_names[4]: "d", col_names[5]: "Correct Answer",
    col_names[6]: "Diagram", col_names[7]: "Year and Board"})
    return df 



def parse_answer(text):
    try:
      match = re.search(r'(\{.*?\})', text)
      answer_text = match.group(1)
      final = json.loads(answer_text)['Answer']
      return final
    except:
        print("Error at extracting answer")
        return "ParsingError" + text