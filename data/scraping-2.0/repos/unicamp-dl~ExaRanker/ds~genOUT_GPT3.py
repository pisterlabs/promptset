# generate GPT3 output from MSMARCO dataset
import os
import csv
from utils import MyUtils
import openai
import json

os.system('clear')


#control flags
#rodar verboso
verbose = 0
#rodar a gpt via api da openai
run_gpt3 = 1
#demo
demo =0

if demo==1:
    file_run = "ds/dsgpt3demo.tsv"
else:
    file_run = "ds/dsgpt3.tsv"


openai.api_key = 'key'

id = 0

func = MyUtils()

with open(file_run, encoding='utf8') as filein:
    f = csv.reader(filein, delimiter="\t")
        
    for line in f:
        label = line[0]
        query = line[1]
        doc = line[2]

        prompt_txt = func.gen_prompt(label, query, doc)
            
        if verbose ==1:
            print('### ' + str(id))
            print(label)
            print(query)
            print(doc)
            print(prompt_txt)
        
        if run_gpt3 ==1:
            output = openai.Completion.create(
                engine='text-davinci-002',
                prompt=prompt_txt,
                max_tokens=256,
                temperature=0.0,
                top_p=0.0,
                stop='',
                logprobs=1,
                echo=False)['choices'][0]

            jsonf = json.dumps(output)
            
            fj = open("ds/out_gpt3/output" + str(id) + ".json","w")
            fj.write(jsonf)
            fj.close()
            
            if verbose ==1:
                print('Output:')
                print(output)
            
            print(str(id))

        id = id +1


