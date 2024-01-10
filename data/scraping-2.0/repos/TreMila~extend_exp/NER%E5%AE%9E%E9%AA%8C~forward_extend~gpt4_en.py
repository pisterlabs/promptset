import openai
from tqdm import tqdm
import time

openai.api_key = 'api_key'

with open('../ACE05/processed_data/train_input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/ace05/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]

with open('./data/ace05/merged_golds.txt','r',encoding='utf-8') as f:
    merged_golds = [eval(line) for line in f]


def sample(merged_golds, type):
    new_merged_golds = []
    for golds in merged_golds:
        tmp = []
        for item in golds:
            if item[1]==type:
                tmp.append(item)
        if len(new_merged_golds) == 100:
            break
        if tmp:
            new_merged_golds.append(tmp)

    return new_merged_golds


def get_sys_prompt():
    return(f'''You are currently a senior entity type generation expert.

Your task is to generate the entity type for the entity span in the given text, following the rules when generating:
1. Based on the entity span, ie (ischemic stroke), combined with the context of the given text, the possible entity types are generated
2. When an entity span is given, an entity type will be generated correspondingly in sequence

Here is an example thought process to guide you in solving the above problems step by step:
Input: ""Our president has put homeland security in the hands of failed Republican hacks."Entity span: [Republican, Our president]"
Output: [organization, person]''')


def get_forward_prompt(example, span_list):
    return(f'''Input : ""{example}" Entity span: {span_list}"''')


def call_chatgpt(span_list,example,p):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt()},
            {"role": "user", "content": get_forward_prompt(example, span_list)},
        ]
    )
    return completion.choices[0].message.content


def run(input_list, merged_golds, mode):
    for idx, golds in tqdm(enumerate(merged_golds), total=len(merged_golds), desc="Processing..."):
        example = input_list[golds[0][0]]
        span_list = [gold[2] for gold in golds]
        
        span_list_str = '['
        for span in span_list:
            span_list_str += span + ', '
        span_list_str = span_list_str[:-2] + ']'

        gold_type_list = [gold[1] for gold in golds]

        while 1:
            try:
                rsp_1 = call_chatgpt(span_list_str, example, 0.3)
                # rsp_2 = call_chatgpt(span_list_str, example, 0.6)
                # rsp_3 = call_chatgpt(span_list_str, example, 1)

                forward_details_path = f'./results/extend_forward_{mode}.txt'
                with open(forward_details_path, 'a', encoding='utf-8') as f:
                    f.write(str(idx+21) + "：\n" + rsp_1 + "\n" + "gold:" + str(gold_type_list) + "\n\n")

                break
            except Exception as e:
                print(e)
                if 'That model is currently overloaded with other requests' in e.user_message:
                    print("resting\n")
                    time.sleep(30)

merged_golds = sample(merged_golds, 'person')
run(input_list, merged_golds[20:40],'gpt4_en')