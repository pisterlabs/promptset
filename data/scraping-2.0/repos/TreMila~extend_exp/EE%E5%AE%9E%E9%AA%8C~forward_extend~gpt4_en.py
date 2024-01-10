import openai
from tqdm import tqdm
import time

openai.api_key = 'api_key'

with open('../CASIE/processed_data/train_input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/casie/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]


def sample(golds, type):
    new_golds = []
    for item in golds:
        if item[1]==type:
            new_golds.append(item)
    return new_golds


def get_sys_prompt():
    return(f'''You are currently a senior event type detection expert.

Your task is to generate event types for event trigger words in a given text, following the rules when generating:
1. Based on the event trigger word, namely (bankruptcy), combined with the context of the given text, the possible event types are generated
2. The output form is: [event type]

The following is an example of a chain of thought to help you think step by step
Input : ""Two of my nephews were excluded from their brother ' s wedding this weekend because they are not LDS."  Trigger word: [wedding]"
Output : [marry]''')


def get_forward_prompt(example, trigger_span):
    return(f'''Input : ""{example}" Trigger word: [{trigger_span}]"''')


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

golds = sample(golds, 'data breach')
run(input_list, golds[:20], 'gpt4_en')