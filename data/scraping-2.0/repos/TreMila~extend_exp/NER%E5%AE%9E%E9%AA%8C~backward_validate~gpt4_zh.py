import openai
import json
from tqdm import tqdm
import time


openai.api_key = 'api_key'


with open('../CMeEE-V2/processed_data/dev_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/cmeee/input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/cmeee/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]

with open('./data/cmeee/merged_golds.txt','r',encoding='utf-8') as f:
    merged_golds = [eval(line) for line in f]

with open('./data/cmeee/final_entity_extend_map_zh.json','r',encoding='utf-8') as f:
    entity_extend_map = json.load(f)

with open('./data/cmeee/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]

entity_type_list = ['dru', 'bod', 'pro', 'sym', 'equ', 'ite', 'dep', 'mic', 'dis']
entity_type_dict = {
            'dru':'药物',
            'bod':'身体',
            'pro':'医疗程序',
            'sym':'临床表现',
            'equ':'医疗设备',
            'ite':'医学检验项目',
            'dep':'科室',
            'mic':'微生物类',
            'dis':'疾病'
}

input_list2ent={}
for idx, item in enumerate(input_list):
    input_list2ent[item] = entity_type_list[idx//10]

id2dev_input_list = {}
for idx, item in enumerate(dev_input_list):
    id2dev_input_list[idx] = item

def get_sys_prompt(cot):
    return(f'''当前你是一个资深的实体提取的专家。

你的任务是给定文本和实体类型type，提取符合实体类型的实体span，在生成的时候遵守以下的规定

1. 基于给定的实体类型type，联合给定文本的上下文，提取可能存在给定实体类型的实体span
2. 基于成对的实体类型type和实体span，生成判断句，并检测每一个判断句是否正确，仅输出“是”或“否”
3. 根据判断句生成实体列表，（实体类型type，实体span），其中实体类型type必须是给定实体类型type

以下是思维链的例子来帮助你一步一步的思考，从而解决上面的问题
Input : {id2dev_input_list[cot[0]]}entity type：[{entity_type_dict[cot[1]]}]
实体span: [（{cot[2]}）]
answer：
{cot[2]}是{entity_type_dict[cot[1]]}吗？是

生成实体列表：
``
（{entity_type_dict[cot[1]]}， {cot[2]}）
``''')


def get_backward_prompt(example, extend_ent):
    return(f'''Input : {example} entity type：[{extend_ent}]''')


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


def run(input_list, input_list2ent, entity_extend_map, cots, mode):
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

run(input_list[80:90], input_list2ent, entity_extend_map, cots[80:90], 'gpt4_zh')
run_gold(input_list, input_list2ent, cots,'chatgpt_zh')