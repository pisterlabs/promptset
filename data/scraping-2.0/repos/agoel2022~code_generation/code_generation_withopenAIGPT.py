!pip install huggingface_hub text_generation requests langchain openai

import json
import os
import shutil
import requests

import openai

from huggingface_hub import Repository
from text_generation import Client

import requests
import re
import json

import pandas as pd

class generate_code():

  api_key=''
  #api_key=''
  openai.api_key=api_key

  def __init__(self,df):
    self.df=df
    self.num_rows=len(df)
    self.num_columns=len(df.columns)
    self.data_types=df.dtypes

  def query(self,payload):
    response=openai.Completion.create(
      engine="text-davinci-002",
      prompt=payload["inputs"],
      temperature=payload["temperature"],
      max_tokens=payload["max_new_tokens"])

    return response['choices'][0]['text'].strip()


  def prompt_generation(self,question):
    columns=self.df.columns
    col_list=','.join(columns)
    prompt=f'''Write a python code on pandas dataframe df with columns:{col_list}.
    The pandas dataframe already exists with the following schema:
    {self.df.dtypes}
    The code assumes the dataframe df while the below code.
    The code should be able to display results for the following user query:
    {question}
    The code should also print the final results using the print statement
    'Do not add comments in the code'
    The code must be executed when passed to exec function
    'Always assume that dataframe df already exists'
    'The code should import the necessary modules'
    'Try to avoid using loops while writing the code if possible'
'''
    return prompt

  @staticmethod
  def clean_code(code):
    read_patt=re.compile(r'pd\.read\_csv\([A-Za-z0-9]*\.csv\)')
    code=re.sub(read_patt,'',code)
    matplotlib_patt=re.compile(r"\%matplotlib inline")
    code=re.sub(matplotlib_patt,'',code)
    return code


  def generate(self,
            original_prompt, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0
        ):

            prompt=original_prompt
            generate_kwargs = {
                "temperature":temperature,
                "max_new_tokens":max_new_tokens,
                "top_p":top_p,
                "repetition_penalty":repetition_penalty,
                "do_sample":True,
                "seed":42
            }

            suffix=''

            prompt_dict={"inputs":prompt,**generate_kwargs}
            code_response=self.query(prompt_dict)
            #for i in range(5):
            prompt_dict={"inputs":prompt,**generate_kwargs}
            code_response=self.query(prompt_dict)
            prompt=code_response

            match = re.search(
              "(```python)(.*)(```)",
               prompt.replace(prompt+suffix, ""),
                re.DOTALL | re.MULTILINE,
                            )
            # if match:
            #   break


            final_code=prompt.replace(original_prompt+suffix, "")
            #final_code=final_code.split("\'''")
            #final_code=final_code.split("You can use the below code to get the answer:")

            return final_code

def ask_csv(df,questions,show_code=False):
  code_generator=generate_code(df)
  prompt=code_generator.prompt_generation(question=questions)
  code=code_generator.generate(prompt)
  code=code_generator.clean_code(code)
  if show_code:
      print(code)
  else:
      pass
  for statements in code.split('\n'):
    exec(statements)


# Sample data
data = {
    'member_id': [1, 2, 3, 4, 5,1, 2, 3, 4, 5],
    'claim_id': [101, 102, 103, 104, 105,106,107,108,109,110],
    'Provider_name': ['Provider A', 'Provider B', 'Provider C', 'Provider D', 'Provider E','Provider A', 'Provider B', 'Provider C', 'Provider D', 'Provider E'],
    'icd10code': ['A00', 'B01', 'C02', 'D03', 'E04','A00', 'B01', 'C02', 'D03', 'E04'],
    'claim_amount': [500.0, 750.0, 600.0, 800.0, 900.0,1000,2000,1500,1200,1100]
}

# Create a Pandas DataFrame
df = pd.DataFrame(data)
ask_csv(df,'Show me the comparison of total claims for each ICD10code using graph',True)
