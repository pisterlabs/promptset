# data collection with GPT-3 to examine effect of examples on performance

# import libraries
import numpy as np
import csv
from datetime import datetime
import os
import openai

# filename to write data to
csv_file = '220830_gpt3_aut_examples_experiment.csv' 

# load API key from environment variable 
openai.api_key = os.getenv("OPENAI_API_KEY")

# fixed parameters to supply each run
max_tokens = 200
n = 1 
engine = "text-davinci-002"
presence_penalty = 1
frequency_penalty = 1 

# GPT-3 variables and parameters to optimize
temperature = np.arange(.6, .81, .05)

# AUT variables
aut_object = ['book', 'fork', 'tin can']
instr_common = 'Think of as many creative uses as you can for a {}. Certainly, there are common, uncreative ways to use a {}. However, for this task, only list creative ideas.'

# AUT instructions per condition 
# condition 1 no examples
instr_0 = '' 
# condition 2 Negative uncreative (low bar) examples: uncreative_examples 
instr_1 = ' Examples of some uncreative ideas are: {}. Now, try to come up with your own creative ideas.'
# condition 3 Negative common (high bar) examples: common_examples
instr_2 = ' Examples of some common, uncreative ideas are: {}. Now, try to come up with your own creative ideas.'
# condition 4 Positive common (low bar) examples: common_examples
instr_3 = ' Examples of some common, creative ideas are: {}. Now, try to come up with your own creative ideas.'
# condition 5 Positive creative (high bar) examples: creative_examples
instr_4 = ' Examples of some creative ideas are: {}. Now, try to come up with your own creative ideas.'

instr_condition = {0: instr_0, 1: instr_1, 2: instr_2, 3: instr_3, 4: instr_4}

# AUT examples per object
examples_0 = {'book': '', 'fork': '', 'tin can': ''}
examples_1 = {'book': 'read, learn something, as a notebook', 'fork': 'eat, stir, as a knife', 'tin can': 'store food,  drink from, as a container'}
examples_2 = {'book': 'bookend, doorstop, raise computer screen', 'fork': 'poke holes in something, brush hair, as a weapon', 'tin can': 'pen holder, tin can telephone, piggybank'}
examples_3 = examples_2
examples_4 = {'book': 'roof tile, table tennis racket, dominos', 'fork': 'tent peg, bend it into a candle holder, stick in wall as coat hanger', 'tin can': 'bridge for ants, showerhead, lampshade'} 
examples = [examples_0, examples_1, examples_2, examples_3, examples_4]

# test combining 3 objects x 5 instructional conditions to get the 15 different prompts
#prompts = []
#for obj in aut_object:
#    for instr_i, instr_val in instr_condition.items():
#        prompts.append(instr_common.format(obj, obj) + instr_val.format(examples[instr_i][obj]) 
#print(prompts)

# function to call gpt3 api with instr and parameter settings
def gpt3_response(eng, instr, max_tok, temp, pres_pnlty, freq_pnlty, num):
    response = openai.Completion.create(
        engine=eng, 
        prompt=instr, 
        max_tokens=max_tok,
        temperature=temp,
        presence_penalty=pres_pnlty,
        frequency_penalty=freq_pnlty,
        n=num)
    return(response)

# open csv file to write data to
f = open(csv_file, 'w')
# create csv writer
writer = csv.writer(f)
# write header to csv file
header = ['id', 'gpt3_id', 'gpt3_created', 'gpt3_model', 'engine', 'timestamp', 'temperature', 'presence_penalty', 'frequency_penalty', 'instr_text', 'instr_cond', 'aut_object', 'gpt3_response']
writer.writerow(header)

# loop through all variables / parameters of the experiment
num_runs = 2 # we want to loop through this twice
rowid = 0
for num in range(num_runs):
    for temp in temperature:
        for aut_obj in aut_object:
            for instr_cond, instr_val in instr_condition.items(): 
                instr = instr_common.format(aut_obj, aut_obj) + instr_val.format(examples[instr_cond][aut_obj])
                response = gpt3_response(engine, instr, max_tokens, temp, presence_penalty, frequency_penalty, n)
                rowid = rowid + 1
                condition_nr = instr_cond + 1 # because we start with 1 when numbering the conditions in human experiment
                gpt3_response_text = response['choices'][0]['text']
                gpt3_id = response['id']
                gpt3_created = response['created']
                gpt3_model = response['model']
                row = [rowid, gpt3_id, gpt3_created, gpt3_model, engine, datetime.now(), temp, presence_penalty, frequency_penalty, instr, condition_nr, aut_obj, gpt3_response_text]
                writer.writerow(row)

# close csv writer and file
f.close()
