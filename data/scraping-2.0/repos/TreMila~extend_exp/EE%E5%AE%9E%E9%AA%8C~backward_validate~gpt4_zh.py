import openai
import json
from tqdm import tqdm
import time


openai.api_key = 'api_key'


with open('../DuEE/processed_data/dev_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('../DuEE/processed_data/labels.txt', 'r', encoding='utf-8') as f:
    event_type_list = [item.strip('\n') for item in f.readlines()]

with open('./data/duee/input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/duee/final_event_extend_map_zh.json','r',encoding='utf-8') as f:
    event_extend_map = json.load(f)

with open('./data/duee/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]


input_list2eve={}
for idx, item in enumerate(input_list):
    input_list2eve[item] = event_type_list[idx//10]

id2dev_input_list = {}
for idx, item in enumerate(dev_input_list):
    id2dev_input_list[idx] = item


def get_sys_prompt(cot):
    return(f'''当前你是一个资深的事件触发词提取专家。

你的任务是给定文本和事件类型type，提取符合事件类型的事件触发词span，在生成的时候遵守以下的规定

1. 基于给定的事件类型type，联合给定文本的上下文，提取可能存在给定事件类型的事件触发词span
2. 基于成对的事件类型type和事件触发词span，生成判断句，并检测每一个判断句是否正确，仅输出“是”或“否”
3. 根据判断句生成事件检测列表，（事件类型type，事件触发词span），其中事件类型type必须是给定事件类型type

以下是思维链的例子来帮助你一步一步的思考，从而解决上面的问题
Input : ""{id2dev_input_list[cot[0]]}" 事件类型type：[{cot[1]}]
事件触发词span: [{cot[2]}]"
answer：
事件类型“{cot[1]}”的触发词是“{cot[2]}”吗？是

生成事件检测列表:
``
（{cot[1]}， {cot[2]}）
``''')


def get_backward_prompt(example, extend_eve):
    return(f'''Input : ""{example}" 事件类型type：[{extend_eve}]"''')


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


def run(input_list, input_list2eve, event_extend_map, cots, mode):
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



run(input_list[20:30], input_list2eve, event_extend_map, cots[20:30], 'gpt4_zh')
