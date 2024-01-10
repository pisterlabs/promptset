import json
from lib2to3.pgen2.token import NOTEQUAL
from os.path import join
from typing import List

from sklearn.metrics import precision_recall_fscore_support

from ee_data import EEDataloader
import openai
import os
import random
import time
import re

from collections import defaultdict

entity_type_dict = {'疾病': 'dis', '临床表现': 'sym', '医疗程序': 'pro', '医疗设备': 'equ', '药物': 'dru', '医学检验项目': 'ite',
                    '身体': 'bod', '科室': 'dep', '微生物类': 'mic'}

reverse_entity_type_dict = {v: k for k, v in entity_type_dict.items()}

def create(**args):
    global all_keys
    openai.api_key = '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'

    try:
        result = openai.ChatCompletion.create(**args)
    except openai.error.RateLimitError:
        result = create(**args)

    return result


def chat(mess):
    # openai.proxy = 'http://127.0.0.1:10809' # 根据自己服务器的vpn情况设置proxy；如果是在自己电脑线下使用，可以在电脑上开vpn然后不加此句代码。
    openai.api_base = "https://closeai.deno.dev/v1"  # 或者利用反向代理openai.com（代理获取：https://github.com/justjavac/openai-proxy）（注释掉上面那句代码）
    responde = create(
        model="gpt-3.5-turbo",
        messages=mess
    )

    res = responde['choices'][0]['message']['content']
    return res


def get_examples(cblue_root='../data/CBLUEDatasets'):

    examples = EEDataloader(cblue_root, augment=False).get_data("train")

    idx = random.randint(0, len(examples) - 1)
    text = examples[idx].text
    entity_list = examples[idx].entities
    prompt_text = f"{text} \n"
    entity_divided_by_type = defaultdict(list)

    for _, entity_dict in enumerate(entity_list):
        start_idx = entity_dict['start_idx']
        end_idx = entity_dict['end_idx']
        entity_type = reverse_entity_type_dict[entity_dict['type']]
        entity = entity_dict['entity']
        prompt_entity = f"|实体类别|实体名称|实体起始位置|实体结束位置|\n|-----|-----|-----|-----|\n|{entity_type}|{entity}|{start_idx}|{end_idx}|\n"
        entity_divided_by_type[entity_type].append(prompt_entity)

    return prompt_text, entity_divided_by_type


def chat_ner(sentence, chatbot):
    mess = [{"role": "system", "content": "You are a helpful assistant."}]
    entity_type_list = ["疾病", "临床表现", "医疗程序", "医疗设备", "药物", "医学检验项目", "身体", "科室", "微生物类"]
    s0p = '''下为一个医学领域命名实体识别的例子，你可以根据这个例子来回答下面的问题。\n\n给定的句子为："{}"\n\n给定实体类型列表：{}\n\n
                在这个句子中，可能包含了哪些实体类型？\n如果不存在则回答：无\n\n'''

    s0p_plus = '''根据给定的句子，请识别出其中类型是"{}"的实体。\n如果不存在则回答：无\n按照表格形式回复，表格有四列且表头为（实体类型，实体名称，实体起始位置，实体结束位置）（注：位置从零开始计数）：'''

    s1p = '''样例到此结束，以下是你需要正式回答的问题：\n给定的句子为："{}"\n\n给定实体类型列表：{}\n\n在这个句子中，可能包含了哪些实体类型？\n如果不存在则回答：无\n
                按照元组形式回复，如 (实体类型1, 实体类型2, ……)：'''.format(sentence, str(entity_type_list))
    ner_s2_p = '''根据需要你回答的给定的句子，请识别出其中类型是"{}"的实体。\n如果不存在则回答：无\n按照表格形式回复，表格有四列且表头为（实体类型，实体名称，实体起始位置，实体结束位置）（注：位置从零开始计数）：'''
    out = {}
    entities = []
    out['text'] = sentence
    # print(s1p)

    example_text, entity_divided_by_type = get_examples()

    mess.append({"role": "user", "content": s0p.format(sentence, str(entity_type_list))})
    mess.append({"role": "assistant", "content": "(" + ",".join(list(entity_divided_by_type.keys())) + ")"})
    for key, value in entity_divided_by_type.items():
        mess.append({"role": "user", "content": s0p_plus.format(key)})
        mess.append({"role": "assistant", "content": '\n\n'.join(value)})
    print(mess)

    mess.append({"role": "user", "content": s1p})
    text1 = chatbot(mess)
    mess.append({"role": "assistant", "content": text1})
    # print(text1)

    res1 = re.findall(r"\(.*?\)", text1)
    # print(res1)

    if res1 != []:
        rels = [tmp[1:-1].split(",") for tmp in res1]
        rels = list(set([re.sub('[\'"]', '', j).strip() for i in rels for j in i]))
    else:
        rels = []
    # print(rels)

    for r in rels:
        if r in entity_type_list:
            s2p = ner_s2_p.format(r)
            # print(s2p)

            mess.append({"role": "user", "content": s2p})
            text2 = chatbot(mess)
            mess.append({"role": "assistant", "content": text2})
            # print(text2)

            res2 = re.findall(r"\|.*?\|.*?\|.*?\|.*?\|", text2)
            # print(res2)

            count = 0
            for so in res2:
                count += 1
                if count <= 2:
                    continue
                so = so[1:-1].split("|")
                so = [re.sub('[\'"]', '', i).strip() for i in so]
                if len(so) == 4:
                    s, o, on, off = so
                    try:
                        entities.append({'entity': o, 'type': entity_type_dict[r], 'start_idx': int(on), 'end_idx': int(off)})
                    except:
                        continue
    out['entities'] = entities

    return out, mess


def generate_test_results(cblue_root='../data/CBLUEDatasets'):
    test_examples = EEDataloader(cblue_root, augment=False).get_data("select_dev")[:1]
    output_dir = "../ckpts/chatgpt_api"
    final_answer = []

    for i, example in enumerate(test_examples):
        sentence = example.text
        print(f'number {i} sentence:', sentence)
        out, mess = chat_ner(sentence, chat)

        print(out)
        final_answer.append(out)
        print("\n")

    with open(join(output_dir, "CMeEE_Dev_Select_ChatIE.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    generate_test_results()




