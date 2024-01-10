import openai
import json
from tqdm import tqdm
import time


openai.api_key = 'api_key'

with open('../CASIE/processed_data/dev_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('../CASIE/processed_data/labels.txt', 'r', encoding='utf-8') as f:
    event_type_list = [item.strip('\n') for item in f.readlines()]

with open('./data/casie/input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/casie/final_event_extend_map_en.json','r',encoding='utf-8') as f:
    event_extend_map = json.load(f)

with open('./data/casie/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]

input_list2eve={}
for idx, item in enumerate(input_list):
    input_list2eve[item] = event_type_list[idx//10]

id2dev_input_list = {}
for idx, item in enumerate(dev_input_list):
    id2dev_input_list[idx] = item


def get_sys_prompt(cot):
    return(f'''You are currently an expert in extracting event-triggered words.

Your task is to extract the trigger words that match the event type given the text and event type, and follow the following rules when generating

1. Based on the given event type, combined with the context of the given text, extract the trigger word span that may exist in the given event type
2. Generate judgment sentences based on the paired event type and event trigger word span, and check whether each judgment sentence is correct, and only output "yes" or "no"
3. Generate an event detection list according to the judgment sentence, (event type, event trigger span), where the event type must be a given event type

The following is an example of a chain of thought to help you think step by step to solve the above problems
Input : ""{id2dev_input_list[cot[0]]}" Event type: [{cot[1]}]
Trigger span: [{cot[2]}]"
answer：
Is the trigger word for event type "{cot[1]}" "{cot[2]}"? yes

Generate a list of event detections:
``
({cot[1]}, {cot[2]})
``''')


def get_backward_prompt(example, extend_eve):
    return(f'''Input: ""{example}" Event type: [{extend_eve}]"''')


def call_chatgpt(example,extend_eve,cot,p):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt(cot)},
            {"role": "user", "content": get_backward_prompt(example, extend_eve)},
        ]
    )

    return completion.choices[0].message.content


def run(input_list, input_list2eve, event_extend_map, cots,mode):
    for idx, example in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        e_type = input_list2eve[example]
        eve_extend = event_extend_map[e_type]
        cot = cots[idx]

        for eve in eve_extend:
            while 1:
                try:
                    rsp_1 = call_chatgpt(example, eve, cot, 0.3) 
                    # rsp_2 = call_chatgpt(example, eve, cot, 0.6)
                    # rsp_3 = call_chatgpt(example, eve, cot, 1)

                    backward_details_path = f'./results/details_{mode}.txt'
                    with open(backward_details_path, 'a', encoding='utf-8') as f:
                        f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\n\n")
                    break
                except Exception as e:
                    print(e)
                    if 'That model is currently overloaded with other requests' in e.user_message:
                        print("resting\n")
                        time.sleep(30)
        


run(input_list[10:20], input_list2eve, event_extend_map, cots[10:20],'gpt4_en')
