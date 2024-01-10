# %%
#再帰的に構造ー物性相関データセットから理由を生成させるバッチ処理

#ライブラリの自動インポート

# %%
import openai
from tqdm import tqdm
import pandas as pd
import os
import glob
import json
import copy
import random

import re

# %%

#環境構築からapi keyを読み込み
openai.api_key =os.environ["OPENAI_API_KEY"]

# %%
#laod dataset as dict
csv_path="dataset/BradleyMeltingPointDataset_clean.csv"
df=pd.read_csv(csv_path)
chemical_records=df.to_dict(orient='records')
random.shuffle(chemical_records)
chemical_records[:1]

# %%
model="gpt-4-1106-preview"

# %%
system_prompt="""
Provide the quantitative Reason and Prediction so that a scientist, who does not know the melting point, can predict the value.

#Commands
- You must quantitatively consider how the melting point shifts, focusing on each functional groups.
- Actual value and Prediction must match each other.
- If Actual value and Prediction differ each other, rethink Reason.
- If Prediction does not contain numbers for each functional group effect, rethink Reason

#Example reason
- Target compound: Toluene
- Basic unit, benzene has a boiling point of 80.
- Methyl group: +30 (due to larger molecular weight)
- Prediction: 110

"""

# %%
def gen_prompt(chemical_record,reason="",prediction=""):
    name=chemical_record["name"]
    smiles=chemical_record["smiles"]
    value=chemical_record["mpC"]
    prompt=f"""
#Data
-Name: {name}
-SMILES: {smiles} 
-Actual value: {value}
-Reason: {reason}
-Prediction: {prediction}

#Output (JSON keys)
- Reason, Prediction
"""
    return prompt


# %%
import json

#ask gpt
def json_generate(prompt,model="gpt-3.5-turbo-1106"):
    response = openai.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"""{prompt}"""
        }  
    ],
    logprobs=True,
    response_format={ "type": "json_object" }
    )

    return (json.loads(response.choices[0].message.content)),response

def parse_log_probs(response):
    log_probs=[]
    for token_pack in (response.choices[0].logprobs.content):
        token=token_pack.token
        logprob=token_pack.logprob
        log_probs.append((token,logprob))

    return log_probs

#parse prediction
def prediction_string_to_number(prompt,model="gpt-3.5-turbo-1106"):
    response = openai.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": """Extract integer from prediction. Use average if multiple numbers are included.
            Examples:
            In: 70.2 - 75.2 degrees Celsius
            Out: 73
            In: 75.2 degrees Celsius
            Out: 73
            In: For 1-naphthalenecarboxaldehyde, starting with the base value for naphthalene with a melting point of 80\u00b0C and subtracting the estimated aldehyde effect of approximately -47 to -50\u00b0C, the predicted melting point would be in the range of 30-33\u00b0C.
            Out: 32
            """,
        },
        {
            "role": "user",
            "content": f"""{prompt}
"#Output (JSON keys)
- Prediction"""
        }  
    ],
    response_format={ "type": "json_object" }
    )

    return (json.loads(response.choices[0].message.content))

# %%

save_base_path="dataset/231225AutoReasoning/"


#load finished records
gen_records={}
gen_json_path_list=glob.glob(save_base_path+"*.json")
for gen_json_path in tqdm(gen_json_path_list):
    with open(gen_json_path) as f:
        gen_hist=json.load(f)
    gen_records[gen_hist[0]["name"]]=gen_hist

# %%

def remove_non_alphabet_characters(s):
    # Using regex to remove all non-alphabet characters
    return re.sub('[^a-zA-Z]', '', s)

# %%
n_recursion=2
n_random_repeat=3
error_threshold=10

# %%
import time

#batch 
for chemical_record in tqdm(chemical_records):

    #load record
    gen_record=copy.deepcopy(chemical_record)

    #skip if already generated
    if gen_record["name"] in gen_records:
        print(f"Skip because already generated: {gen_record['name']}")
        continue

    record_history=[]

    fin_flag=False
    #make suggestion with random seed
    for j in range(n_random_repeat):
        if fin_flag:
            break

        gen_record["Reason"]=""
        gen_record["Prediction"]=""
        if j==0:
            record_history.append(copy.deepcopy(gen_record))

        #improve reasoing
        for i in range(n_recursion):
            try:
                r,response=json_generate(
                    gen_prompt(gen_record,
                            reason=gen_record["Reason"],
                            prediction=gen_record["Prediction"]
                    ),
                    model=model,
                )
                r["logprob"]=parse_log_probs(response)
                time.sleep(1)
            except Exception as e:
                #error occurs especially when generatin JSON data by gpt & rate limit
                print(e)
                fin_flag=True
                time.sleep(3600*5)
                break
            #parse prediction string to number
            gen_record.update(r)
            try:
                gen_record["Prediction(integer)"]=float(prediction_string_to_number(gen_record["Prediction"])["Prediction"])
            except:
                gen_record["Prediction(integer)"]=99999
            record_history.append(copy.deepcopy(gen_record))
            
            #finish reasoning if prediction is close to actual value
            if abs(gen_record["Prediction(integer)"]-gen_record["mpC"])<=error_threshold:
                fin_flag=True
                print(f"Finished because good reasoning was achieved: {gen_record['name']}")
                break

    #save
    save_name=remove_non_alphabet_characters(gen_record["name"])
    save_path=save_base_path+f"{save_name}.json"
    with open(save_path, 'w') as f:
        json.dump(record_history, f, indent=4)
    print(save_path)

    gen_records[gen_record["name"]]=record_history


