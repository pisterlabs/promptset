#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 20:49:57 2023

@author: alon
"""
import pandas as pd 
import os
import openai
import wandb
import json
import time

# In[]
#OPENAI_API_KEY="sk-bMmb98t2zXgn9cx5FTSpT3BlbkFJDEkYbjDE4Hcgz3ilMnA"

#os.environ[‘OPENAI_API_KEY’] = ‘your key’

openai.api_key = os.getenv("OPENAI_API_KEY")
with open(r"C:\Users\Ravit\Documents\rnd\horizon_scanning_lab\feature_scripts\output_htmls\13Feb23_new_sites_html_text.json") as f:
    dict_htmls = json.loads(f.read())

with open(r"C:\Users\Ravit\Documents\rnd\horizon_scanning_lab\feature_scripts\output_pdfs\13Feb23_arxiv_pdf_text.json") as f:
    dict_pdfs = json.loads(f.read())
    


# In[]
df_html_gpt = pd.DataFrame([])



# In[]
def extract_terms_with_gpt(text):
    
    gpt_prompt = "Find in the text new emerging technologies." + text
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=gpt_prompt,
        temperature=0.5,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )
    
    return response['choices'][0]['text']

# In[]
html_indices = dict_htmls.keys()
pdf_indices = dict_pdfs.keys()

htmls_terms = []
pdfs_terms = []

for html_index in html_indices:
     htmls_terms.append(extract_terms_with_gpt(dict_htmls[html_index])) 
     time.sleep(3)
###  pdfs are too long
#     pdfs_terms.append(extract_terms_with_gpt(dict_pdfs[pdf_index])) 

# In[]   
    
df_html_gpt['gpt3_tech_extraction'] = htmls_terms
df_html_gpt.index = list(html_indices)[:len(htmls_terms)]

# In[]
for i, row in df_html_gpt.iterrows():
    df_html_gpt.loc[i] = row['gpt3_tech_extraction'].replace('\n', '')

# In[]
df_html_gpt.to_csv(r'C:\Users\Ravit\Documents\rnd\horizon_scanning_lab\feature_scripts\docs\gpt_output\13Feb23_gpt_term_extraction.csv')


# In[]
'''
run = wandb.init(project='GPT-3 in Python')
prediction_table = wandb.Table(columns=["prompt", "completion"])

prediction_table.add_data(gpt_prompt,response['choices'][0]['text'])
wandb.log({'predictions': prediction_table})
wandb.finish()
'''