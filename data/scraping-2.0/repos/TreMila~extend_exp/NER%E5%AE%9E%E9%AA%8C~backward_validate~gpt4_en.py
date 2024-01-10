import openai
import json
from tqdm import tqdm
import time

openai.api_key = 'api_key'


with open('../ACE05/processed_data/dev_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/ace05/input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/ace05/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]

with open('./data/ace05/merged_golds.txt','r',encoding='utf-8') as f:
    merged_golds = [eval(line) for line in f]

with open('./data/ace05/final_entity_extend_map_en.json','r',encoding='utf-8') as f:
    entity_extend_map = json.load(f)

with open('./data/ace05/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]

with open('../ACE05/labels.json','r',encoding='utf-8') as f:
    entity_type_list = json.load(f)

input_list2ent={}
for idx, item in enumerate(input_list):
    input_list2ent[item] = entity_type_list[idx//10]

id2dev_input_list = {}
for idx, item in enumerate(dev_input_list):
    id2dev_input_list[idx] = item

def get_sys_prompt(cot):
    return(f'''You are an experienced senior expert in entity extraction in the field of IT.

Your current task is to extract the specific entity span that corresponds to the given entity types and  context. To accomplish this, please adhere to the following guidelines:

1. Extract entity spans that may correspond to the given entity type by combining the context of the given text.
2. Generate judgment sentences based on paired entity types and entity spans. Evaluate the correctness of each judgment sentence and output only 'yes' or 'no'.
3. Generate an entity list based on the judgment sentences, following the format (entity type, entity span). Ensure that the entity type matches the given entity type.

Here is an example thought process to guide you in solving the above problems step by step:
Input : ""{id2dev_input_list[cot[0]]}" Entity Type: [{cot[1]}]"
Entity Span: [{cot[2]}]
answer：
Is '{cot[2]}' a {cot[1]}? yes

Generate an entity list:
``
({cot[1]}, {cot[2]})
``''')


def get_backward_prompt(example, extend_ent):
    return(f'''Input : ""{example}" Entity type: [{extend_ent}]"''')


def call_chatgpt(example,extend_ent,cot,p):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt(cot)},
            {"role": "user", "content": get_backward_prompt(example, extend_ent)},
        ]
    )
    return completion.choices[0].message.content


def run(input_list, input_list2ent, entity_extend_map, cots,mode):
    for idx, example in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        ent = input_list2ent[example]
        ent_extend = entity_extend_map[ent]
        cot = cots[idx]

        for ent in ent_extend:
            while 1:
                try:
                    rsp_1 = call_chatgpt(example, ent, cot, 0.3) 
                    # rsp_2 = call_chatgpt(example, ent, cot, 0.6)
                    # rsp_3 = call_chatgpt(example, ent, cot, 1)

                    backward_details_path = f'./results/details_{mode}.txt'
                    with open(backward_details_path, 'a', encoding='utf-8') as f:
                        f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\n\n")
                    break
                except Exception as e:
                    print(e)
                    if 'That model is currently overloaded with other requests' in e.user_message:
                        print("resting\n")
                        time.sleep(30)


def run_gold(input_list, input_list2ent,cots,mode):
    for idx, example in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        ent = input_list2ent[example]
        cot = cots[idx]

        while 1:
            try:
                rsp_1 = call_chatgpt(example, ent, cot, 0.3) 
                rsp_2 = call_chatgpt(example, ent, cot, 0.6)
                rsp_3 = call_chatgpt(example, ent, cot, 1)

                backward_details_path = f'./results/details_gold_{mode}.txt'
                with open(backward_details_path, 'a', encoding='utf-8') as f:
                    f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\nOutput:" + rsp_2 + "\nOutput:" + rsp_3 + "\n\n")

                break
            except Exception as e:
                print(e)
                if 'That model is currently overloaded with other requests' in e.user_message:
                    print("resting\n")
                    time.sleep(30)

run(input_list[10:20], input_list2ent, entity_extend_map, cots[10:20],'gpt4_en')
run_gold(input_list, input_list2ent, cots,'chatgpt_en')