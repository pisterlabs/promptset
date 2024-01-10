from tqdm import tqdm, trange
import random
import sys
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("facebook/bart-large")

testfile_name=sys.argv[1] # 예제 : wp_all_generations_outputs
korean=int(sys.argv[2])
debug=int(sys.argv[3]) # 1 or 0
PAR_COUNT=int(sys.argv[4]) # 10
if debug==1:
    debug=True
else:
    debug=False
if korean==1:
    korean=True
else:
    korean=False

print("test file : " + testfile_name + ".csv")
print("debug mode : " + str(debug))


import csv
import ctypes as ct
import math
import numpy as np
import pandas as pd
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

_f = pd.read_csv(testfile_name+'.csv',chunksize=1000)
_f= pd.concat(_f)

num_whole_steps=len(_f.index)

first=True

count=0
par_count=0
last_keywords=""
cumul_fake_outputs=""
cumul_real_outputs=""



f=[]
r=[]
step=0
is_real_nan=False
is_real_nan_cumul=False
#progress_bar = tqdm(range(num_whole_steps))

for step, line in _f.iterrows():
    
    #if first:
    #    first=False
    #    continue
    count+=1
    #progress_bar.update(1)
    #print(line[2])
    if line[3]=="real text" or line[4]=="generated_results":
        print("pass this line")
        continue

    if line[3]!=line[3]:
        is_real_nan=True
    keywords=line[2].replace('[','').replace(']','')
    fake=line[4]
    #fake=line[4].replace('[','').replace(']','').replace('<','').replace('>','').replace('newline','').replace('Newline','').replace('ewline','').replace('new line','').replace('ew line','').replace('\\x90','').replace("\\x80",'').replace('\\x9c','').replace('\\x84','').replace("\\x9d",'').replace('\\x99','').replace('\\x9','').replace('\\x8','')
    if is_real_nan is False:
        real=line[3]
        #real=line[3].replace('[','').replace(']','').replace('<','').replace('>','').replace('newline','').replace('Newline','').replace('ewline','').replace('new line','').replace('ew line','').replace('\\x90','').replace('\\x80','').replace('\\x9c','').replace('\\x84','').replace("\\x9d",'').replace('\\x99','').replace('\\x9','').replace('\\x8','')
    else:
        real=""

    if keywords==last_keywords:
        cumul_fake_outputs+=fake+"\n"
        cumul_real_outputs+=real+"\n"
        if is_real_nan:
            is_real_nan_cumul=True
        is_real_nan=False
        par_count+=1

        continue
    else:
        if count!=1 and par_count<PAR_COUNT:
            if cumul_fake_outputs==cumul_fake_outputs and cumul_real_outputs==cumul_real_outputs:
                f.append({"text" : cumul_fake_outputs.encode('ascii','ignore').decode('ascii',errors='ignore') , "label" : "fake"})
                if is_real_nan_cumul is False:
                    r.append({"text" : cumul_real_outputs.encode('ascii','ignore').decode('ascii',errors='ignore') , "label" : "real"})
        
        is_real_nan_cumul=False
        par_count=0
        cumul_fake_outputs=fake
        cumul_real_outputs=real
        last_keywords=keywords


f.append({"text" : cumul_fake_outputs , "label" : "fake"})
r.append({"text" : cumul_real_outputs, "label" : "real"})
step+=1
random.shuffle(f)
random.shuffle(r)
mix=f+r
random.shuffle(mix)
random.shuffle(mix)
import os
import openai
print("mixing done.")
print(len(mix))

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error Creating directory. ' + directory)
createFolder('HumanEvaluate')
createFolder('HumanEvaluate/TestFiles')
print('HumanEvaluate/TestFiles/'+'/'.join(testfile_name.split('/')[:-1]))
createFolder('HumanEvaluate/TestFiles/'+'/'.join(testfile_name.split('/')[:-1]))

f = open("HumanEvaluate/TestFiles/"+testfile_name +'.csv','w', newline='')
wr = csv.writer(f)
total_tokens=0
total_fake=0
openai.api_key = "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"

for i,sample in enumerate(mix[:50]):
    
    input_tokens=tokenizer(sample["text"],return_tensors="pt").input_ids[0]
    
    print(len(input_tokens))
    if korean:
        response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Translate this into Korean :\n\n"+sample["text"],
    temperature=0.3,
    max_tokens=3800-len(input_tokens),
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
        total_tokens+=response["usage"]['total_tokens']
    if "f" in sample["label"]:
        total_fake+=1
    if debug:
        print(sample["text"])
        if korean:
            print(response["choices"][0]["text"])
        print(sample["label"])
        print("total tokens : " + str(total_tokens) + ", so the price was : " + str(total_tokens / 1000 * 0.02) + " dollors.")
        input()
    if korean:
        trans=response["choices"][0]["text"]
    else:
        trans=""
    wr.writerow([i,sample["text"],trans,sample["label"]])
    print(sample["label"])
    print("f" in sample["label"])
    print(total_fake)
    print(total_tokens)
    print("*")

print("total tokens : " + str(total_tokens) + ", so the price was : " + str(total_tokens / 1000 * 0.02) + " dollors.")
print("total_fake : " + str(total_fake) + " total_real : " + str(50-total_fake))
