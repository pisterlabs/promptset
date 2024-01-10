import openai
from tqdm import tqdm
import time

openai.api_key = 'api_key'


with open('./data/cmeie/input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/cmeie/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]

with open(f'./data/cmeie/cot_examples.txt','r',encoding='utf-8') as f:
    cot_examples = [eval(line) for line in f]


def sample(golds, type):
    new_golds = []
    for item in golds:
        if item[1]==type:
            new_golds.append(item)  
        
    return new_golds


def get_sys_prompt(input_list, cot_example):
    return(f'''当前你是一个资深的关系生成专家。你的任务是对给定文本中的主客体对生成关系类型，在生成时遵守以下的规定：基于主客体对，即（缺血性卒中，MRI），联合给定文本的上下文，生成可能存在的关系类型。

以下是思维链的例子来帮助你一步一步的思考
Input : “{input_list[cot_example[0]]}” 主客体对：[（{cot_example[2]}，{cot_example[3]}）]
Output : [{cot_example[1]}] ''')


def get_forward_prompt(example, gold):
    return(f'''Input : “{example}” 主客体对：[（{gold[2]}，{gold[3]}）]''')


def call_chatgpt(input_list, gold, example, cot, p):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt(input_list, cot)},
            {"role": "user", "content": get_forward_prompt(example, gold)},
        ]
    )
    return completion.choices[0].message.content


def run(input_list, golds, cot_examples, mode):
    for idx, gold in tqdm(enumerate(golds), total=len(golds), desc="Processing..."):
        example = input_list[gold[0]]
        cur_cot = [item for item in cot_examples if item[1] == gold[1]]
        while 1:
            try:
                rsp_1 = call_chatgpt(input_list, gold, example, cur_cot[0], 0.3)
                # rsp_2 = call_chatgpt(input_list, gold, example, cur_cot[0], 0.6)
                # rsp_3 = call_chatgpt(input_list, gold, example, cur_cot[0], 1)

                forward_details_path = f'./results/extend_forward_{mode}.txt'
                with open(forward_details_path, 'a', encoding='utf-8') as f:
                    f.write(str(idx+21) + "：\n" + rsp_1 + "\n" + "gold:" + str(gold[1]) + "\n\n")

                break
            except Exception as e:
                print(e)
                if 'That model is currently overloaded with other requests' in e.user_message:
                    print("resting\n")
                    time.sleep(30)


golds = sample(golds, '多发群体')
run(input_list, golds[20:40], cot_examples, 'gpt4_zh')