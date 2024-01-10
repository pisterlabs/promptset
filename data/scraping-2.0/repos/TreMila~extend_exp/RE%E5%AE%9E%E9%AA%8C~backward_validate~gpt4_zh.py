import openai
import json
from tqdm import tqdm
import time

openai.api_key = 'api_key'


with open('./data/cmeie/cot_examples.txt','r',encoding='utf-8') as f:
    cot_examples = [item.strip('\n') for item in f.readlines()]

with open('./data/cmeie/cot_spo.txt', 'r', encoding='utf-8') as f:
    cot_spos = [eval(line) for line in f]

with open('./data/cmeie/input_list.txt', 'r', encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/cmeie/final_rel_extend_map_zh.json','r',encoding='utf-8') as f:
    rel_extend_map = json.load(f)

rel_type_list = list(rel_extend_map.keys())

input_list2rel={}
for idx, item in enumerate(input_list):
    input_list2rel[item] = rel_type_list[idx//10]

def get_sys_prompt(cot_example, cot_spo):
    return(f'''当前你是一个资深的信息提取的专家。你的任务是给定文本和关系，提取符合关系类型的主客体对，在生成的时候遵守以下的规定。

1. 基于给定的关系，联合给定文本的上下文，提取可能存在给定关系的主客体对
2. 基于关系列表和关键词，生成判断句，并检测每一个判断句是否正确，仅输出“是”或“否”
3. 根据判断句生成关系列表，（主体，关系，客体），其中关系必须是给定关系

以下是思维链的例子来帮助你一步一步的思考，从而解决上面的问题
Input : "{cot_example}" relation:[{cot_spo[0]}]

主客体对: [({cot_spo[1]}, {cot_spo[2]})]
answer:
{cot_spo[1]}的{cot_spo[0]}是{cot_spo[2]}吗？是

生成关系列表：
``
({cot_spo[1]}, {cot_spo[0]},{cot_spo[2]})
``''')


def get_backward_prompt(example, extend_rel):
    return(f'''Input :"{example}" relation: [{extend_rel}]''')


def call_chatgpt(example,cot_example,cot_spo,extend_rel,p):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt(cot_example, cot_spo)},
            {"role": "user", "content": get_backward_prompt(example, extend_rel)},
        ]
    )

    return completion.choices[0].message.content


def run(input_list, cot_spos,cot_examples, rel_extend_map, mode):
    for idx, example in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        rel = '多发群体'
        rel_extend = rel_extend_map[rel]
        cot_example = cot_examples[idx//10]
        cot_spo = cot_spos[idx//10]

        for r in rel_extend:
             while 1:
                try:
                    rsp_1 = call_chatgpt(example,cot_example, cot_spo, r, 0.3) 
                    # rsp_2 = call_chatgpt(example, r, cot, 0.6)
                    # rsp_3 = call_chatgpt(example, r, cot, 1)

                    backward_details_path = f'./results/details_{mode}.txt'
                    with open(backward_details_path, 'a', encoding='utf-8') as f:
                        f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\n\n")
                    break
                except Exception as e:
                    print(e)
                    if 'That model is currently overloaded with other requests' in e.user_message:
                        print("resting\n")
                        time.sleep(30)

run(input_list[40:50], cot_spos,cot_examples,rel_extend_map, 'gpt4_zh')