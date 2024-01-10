import openai
import json
from tqdm import tqdm
import time

openai.api_key = 'api_key'


with open('./data/scierc/cot_examples.txt','r',encoding='utf-8') as f:
    cot_examples = [item.strip('\n') for item in f.readlines()]

with open('./data/scierc/cot_spo.txt', 'r', encoding='utf-8') as f:
    cot_spos = [eval(line) for line in f]

with open('./data/scierc/input_list.txt', 'r', encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/scierc/final_rel_extend_map_en.json', 'r', encoding='utf-8') as f:
    rel_extend_map = json.load(f)

rel_type_list = ['HYPONYM-OF',
 'FEATURE-OF',
 'USED-FOR',
 'CONJUNCTION',
 'EVALUATE-FOR',
 'PART-OF',
 'COMPARE']

input_list2rel = {}
for idx, item in enumerate(input_list):
    input_list2rel[item] = rel_type_list[idx//10]


def get_sys_prompt(cot_example, cot_spo):
    return(f'''You are currently a senior expert in information extraction.

Your task is to give a text and a relation, extract subject-object pairs that match the relation. DO NOT output any Note. Follow the below rules when generating:

1. Based on the given relation, combine the context of the given text, and extract the subject-object pairs that may have the given relation
2. Generate judgment sentences based on the relation list and keywords, and check whether each judgment sentence is correct, and only output "yes" or "no"
3. Generate a list of relations based on the judgment sentence, (subject, relation, object), where the relation must be a given relation

The following is an example of a chain of thought to help you think step by step to solve the above problems
Input: "{cot_example}" relation: [{cot_spo[1]}]

subject-object pair: [({cot_spo[2]}, {cot_spo[3]})]
ask: Is the relation between "{cot_spo[2]}" and "{cot_spo[3]}" the "{cot_spo[1]}"? answer: Yes

Generate a list of relations:
``
({cot_spo[2]}, {cot_spo[1]}, {cot_spo[3]})
``''')


def get_backward_prompt(example, extend_rel):
    return(f'''Input: "{example}" relation: [{extend_rel}]''')


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
        rel = 'FEATURE-OF'
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



run(input_list[10:20], cot_spos,cot_examples,rel_extend_map,'gpt4_en')
