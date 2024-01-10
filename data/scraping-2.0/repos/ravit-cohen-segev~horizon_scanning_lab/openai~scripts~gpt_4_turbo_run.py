# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:19:06 2023

@author: Ravit
"""

import openai
import pandas as pd
import os
# In[]
def split_list(lst, chunk_size=60):
    for i in range(0, len(lst), chunk_size):
        yield ", ".join(lst[i:i + chunk_size])

# In[]

# Set your API key here
openai.api_key = os.getenv("OPENAI_API_KEY")

with open(r'C:\Users\Ravit\Documents\rnd\horizon_scanning_lab\tech_list\prompts\16Nov2023_TRL_assessment_prompt.txt', encoding='utf-8') as f:
    prompt_template = f.read()

df = pd.read_excel(r'C:\Users\Ravit\Documents\rnd\horizon_scanning_lab\tech_list\docs\Levi_files\20231023 New Master October 2023.xlsx')
tech_lists = list(split_list(list(df['Tech Level 2'].dropna().unique())))
 

prompts = [prompt_template.replace("{tech_list}", tech_list) for tech_list in tech_lists]
# In[]
gpt_answers = []

for prompt in prompts:
    try:
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",  # Replace with the correct model name
            messages=[{"role": "user", "content": prompt}]
        )
        print(chat_completion['choices'][0]['message']['content'])
    except Exception as e:
        print(f"An error occurred: {e}")
    gpt_answers.append(chat_completion['choices'][0]['message']['content'])
# In[]
# parse answers so it won't contain anything but the table itself

for i, answer in enumerate(gpt_answers):
    lines = answer.splitlines()
    clean_answer = '\n'.join([line for line in lines if '|' in line])
    with open(f'C:\\Users\\Ravit\\Documents\\rnd\\horizon_scanning_lab\\tech_list\\docs\\results\\gpt_4_TRL\\16Non23_TRL_{str(i)}.txt', 'w') as f:
       f.write(clean_answer) 
    
    