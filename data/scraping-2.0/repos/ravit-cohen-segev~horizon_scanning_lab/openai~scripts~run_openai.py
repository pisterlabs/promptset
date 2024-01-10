# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:29:49 2023

@author: Ravit
"""

import pandas as pd 
import os
import openai
import wandb
import json
import time
import ast
import numpy as np

# In[]

API_key = os.environ.get("OPENAI_API_KEY")
# In[]
def extract_terms_with_gpt(prompt):
    
    gpt_prompt = f"""
    {{
      "prompt": "{prompt}",
      "output_format": "json"
    }}
    """

    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=gpt_prompt,
        temperature=0,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
   
        )
    
    return response['choices'][0]['text']


# In[]
prompts = pd.read_csv(r'C:\Users\Ravit\Documents\rnd\horizon_scanning_lab\tech_list\docs\13sep23_metatech_syns_prompts.csv')['prompts'].to_list()

responses = []

# In[]
for prompt in prompts:
    responses.append(extract_terms_with_gpt(prompt)) 