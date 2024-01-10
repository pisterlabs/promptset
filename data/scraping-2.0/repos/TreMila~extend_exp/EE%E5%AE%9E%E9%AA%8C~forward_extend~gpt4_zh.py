import openai
from tqdm import tqdm
import time

openai.api_key = 'api_key'

with open('../DuEE/processed_data/train_input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/duee/golds_2.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]


def sample(golds, type):
    new_golds = []
    for item in golds:
        if item[1]==type:
            new_golds.append(item)
    return new_golds


def get_sys_prompt():
    return(f'''当前你是一个资深的事件类型检测专家。
           
你的任务是对给定文本中的事件触发词生成事件类型，在生成时遵守以下的规定：
1.基于事件触发词，即（缺血性卒中），联合给定文本的上下文，生成可能存在的事件类型
2.只需要输出：[事件类型]

以下是思维链的例子来帮助你一步一步的思考
Input : “雀巢裁员4000人：时代抛弃你时，连招呼都不会打!” 事件触发词span：[裁员]
Output : [组织关系-裁员]''')


def get_forward_prompt(example, trigger_span):
    return(f'''Input : “{example}” 事件触发词span：[{trigger_span}]''')


def call_chatgpt(trigger_span,example,p):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt()},
            {"role": "user", "content": get_forward_prompt(example, trigger_span)},
        ]
    )

    return completion.choices[0].message.content


def run(input_list, golds, mode):
    for idx, gold in tqdm(enumerate(golds), total=len(golds), desc="Processing..."):
        example = input_list[gold[0]]
        gold_type = gold[1]
        trigger_span = gold[2]
        
        while 1:
            try:
                rsp_1 = call_chatgpt(trigger_span, example, 0.3)
                # rsp_2 = call_chatgpt(trigger_span, example, 0.6)
                # rsp_3 = call_chatgpt(trigger_span, example, 1)

                forward_details_path = f'./results/extend_forward_{mode}.txt'
                with open(forward_details_path, 'a', encoding='utf-8') as f:
                    f.write(str(idx+1) + "：\n" + rsp_1 + "\n" + "gold:" + str(gold_type) + "\n\n")
                break
            except Exception as e:
                print(e)
                if 'That model is currently overloaded with other requests' in e.user_message:
                    print("resting\n")
                    time.sleep(30)

golds = sample(golds, '交往')
run(input_list, golds[:20], 'gpt4_zh')