import openai
from tqdm import tqdm
import time
import random

openai.api_key = 'api_key'


with open('./data/scierc/input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/scierc/train_input_list.txt','r',encoding='utf-8') as f:
    train_input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/scierc/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]

with open(f'./data/scierc/cot_examples.txt','r',encoding='utf-8') as f:
    cot_examples = [eval(line) for line in f]


def sample(golds, type):
    new_golds = []
    input_ids = []
    for item in golds:
        if item[1]==type:
            new_golds.append(item)
            if item[0] not in input_ids:
                input_ids.append(item[0])
        if len(input_ids) == 90:
            break
        
    return new_golds


def get_sys_prompt(train_input, cot_example):
    return(f'''You are currently a senior relation generation expert.

Your task is to generate relation types for the subject-object pairs in a given text, following the rules: Based on subject-object pairs, ie (noisy-channel architecture, transformation model), combined with the context of the given text, generate type of relations.

The following is an example of a chain of thought to help you think step by step
Input : "{train_input[cot_example[0]]}" Subject-object pair:[({cot_example[2]}, {cot_example[3]})]
Output : [{cot_example[1]}] ''')


def get_forward_prompt(example, gold):
    return(f'''Input : "{example}" Subject-object pair:[({gold[2]}, {gold[3]})]''')


def call_chatgpt(train_input,gold,example,cot,p):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt(train_input, cot)},
            {"role": "user", "content": get_forward_prompt(example, gold)},
        ]
    )
    return completion.choices[0].message.content


def run(train_input, input_list, golds, cot_examples, mode):
    for idx, gold in tqdm(enumerate(golds), total=len(golds), desc="Processing..."):
        example = input_list[gold[0]]
        other_cot = [item for item in cot_examples if item[1] != gold[1]]
        cur_cot = random.choice(other_cot)

        while 1:
            try:
                rsp_1 = call_chatgpt(train_input, gold, example, cur_cot, 0.3)              
                # rsp_2 = call_chatgpt(train_input, gold, example, cur_cot, 0.6)
                # rsp_3 = call_chatgpt(train_input, gold, example, cur_cot, 1)
    
                forward_details_path = f'./results/extend_forward_{mode}.txt'
                with open(forward_details_path, 'a', encoding='utf-8') as f:
                    f.write(str(idx+21) + "：\n" + rsp_1 + "\n" + "gold:" + str(gold[1]) + "\n\n")

                break
            except Exception as e:
                print(e)
                if 'That model is currently overloaded with other requests' in e.user_message:
                    print("resting\n")
                    time.sleep(30)


golds = sample(golds, 'FEATURE-OF')
run(train_input_list, input_list, golds[20:40], cot_examples, 'gpt4_en')
