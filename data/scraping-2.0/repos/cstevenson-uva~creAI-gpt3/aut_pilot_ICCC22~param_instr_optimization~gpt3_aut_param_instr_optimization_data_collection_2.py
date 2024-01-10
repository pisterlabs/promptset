# Monte-Carlo experiment with GPT-3 on the AUT paperclip and book tasks
# aim is to optimize parameters and instructions for optimal AUT performance by GPT-3
# this script is to gather the data from gpt-3
# then raters score each response with snapshot scoring method
# we'll analyze the data to decide which instructions and the paramater settings are best 

# import libraries
import numpy as np
import csv
from datetime import datetime
import os
import openai

# filename to write data to
csv_file = 'data/220416_gpt3_aut_instr_param_optimization.csv' 

# load API key from environment variable 
openai.api_key = os.getenv("OPENAI_API_KEY")

# fixed parameters to supply each run
max_tokens = 200
n = 1 
engine = "text-davinci-002"

# GPT-3 variables and parameters to optimize
temperature = np.arange(.4, 1, .1)
presence_penalty = np.arange(1, 2.1, .5)
frequency_penalty = np.arange(1, 2.1, .5)

# AUT variables Instructions to try
aut_object = ['paperclip', 'book', 'fork', 'brick', 'tin can']
instr_common = 'List 5 creative uses for a {}. '
instr_1 = '' # we didn't have additional instructions for humans
instr_2 = 'The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different.'
instruction = {'instr_1': instr_1, 'instr_2': instr_2}

# function to call gpt3 api with instr and parameter settings
def gpt3_response(eng, instr, max_tok, temp, pres_pnty, freq_pnty, num):
    response = openai.Completion.create(
        engine=eng, 
        prompt=instr, 
        max_tokens=max_tok,
        temperature=pres_pnty,
        presence_penalty=freq_pnty,
        frequency_penalty=freq_pnty,
        n=num)
    return(response)

# open csv file to write data to
f = open(csv_file, 'w')
# create csv writer
writer = csv.writer(f)
# write header to csv file
header = ['id', 'gpt3_id', 'gpt3_created', 'gpt3_model', 'engine', 'timestamp', 'temperature', 'presence_penalty', 'frequency_penalty', 'instr_nr', 'instr_text', 'aut_object', 'gpt3_response']
writer.writerow(header)

# loop through all variables / parameters of the experiment
rowid = 0
for temp in temperature:
    for pres_pnlt in presence_penalty:
        for freq_pnlt in frequency_penalty:
            for aut_obj in aut_object:
                for instr_key, instr_val in instruction.items():
                    instr = instr_common.format(aut_obj) + instr_val.format(aut_obj) 
                    response = gpt3_response(engine, instr, max_tokens, temp, pres_pnlt, freq_pnlt, n)
                    rowid = rowid + 1
                    gpt3_response_text = response['choices'][0]['text']
                    gpt3_id = response['id']
                    gpt3_created = response['created']
                    gpt3_model = response['model']
                    row = [rowid, gpt3_id, gpt3_created, gpt3_model, engine, datetime.now(), temp, pres_pnlt, freq_pnlt, instr_key, instr, aut_obj, gpt3_response_text]
                    writer.writerow(row)


# close csv writer and file
f.close()
