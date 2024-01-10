import json
from lib2to3.pgen2.token import NOTEQUAL
from os.path import join
from typing import List

from sklearn.metrics import precision_recall_fscore_support, f1_score

# from ee_data import EEDataloader
import openai
import os
import random
import time
import re

openai_key = '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'

entity_type_dict = {'疾病': 'dis', '临床表现': 'sym', '医疗程序': 'pro', '医疗设备': 'equ', '药物': 'dru', '医学检验项目': 'ite',
                    '身体': 'bod', '科室': 'dep', '微生物类': 'mic'}

reverse_entity_type_dict = {'dis': '疾病', 'sym': '临床表现', 'pro': '医疗程序', 'equ': '医疗设备', 'dru': '药物', 'ite': '医学检验项目',
                            'bod': '身体', 'dep': '科室', 'mic': '微生物类'}


def get_completion(mess, model='gpt-3.5-turbo', max_tokens=1024, temperature=0.5):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=mess,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=None,
        n=1,
    )
    return completion


def get_examples(num_examples, cblue_root='../data/CBLUEDatasets'):
    example_list = []
    for i in range(num_examples):

        examples = EEDataloader(cblue_root, augment=False).get_data("train")

        idx = random.randint(0, len(examples) - 1)
        text = examples[idx].text
        entity_list = examples[idx].entities
        prompt_text = f"文本：\n {text} \n"

        for _, entity_dict in enumerate(entity_list):
            start_idx = entity_dict['start_idx']
            end_idx = entity_dict['end_idx']
            entity_type = reverse_entity_type_dict[entity_dict['type']]
            entity = entity_dict['entity']
            prompt_entity = f"|实体类别|实体名称|实体起始位置|实体结束位置|\n|-----|-----|-----|-----|\n|{entity_type}|{entity}|{start_idx}|{end_idx}|\n"
            prompt_text += prompt_entity

        example_list.append(prompt_text)

    return example_list


def predict_ner(openai_key, sentences, examples, num_sentence, model='gpt-3.5-turbo', max_tokens=1024, temperature=0.5,
                ent_type_dic=entity_type_dict):
    openai.api_key = openai_key
    openai.api_base = "https://closeai.deno.dev/v1"

    examples_str = "\n".join(examples)

    mess = [{"role": "system", "content": f"你是一个中文医疗信息处理的助手，你要完成命名实体识别的任务，根据给定的句子，识别出其中的实体，如果不存在则回答：无\n \
             按照表格形式回复，表格有四列且表头为（实体类型，实体名称，实体起始位置，实体结束位置）（注：位置从零开始计数）。给定实体类型列表：{ent_type_dic.keys()}。"},
            {"role": "assistant", "content": examples_str}, {"role": "user", "content": sentences}]

    try:
        completion = get_completion(mess, model, max_tokens, temperature)
    except openai.error.RateLimitError:
        time.sleep(20)
        completion = get_completion(mess, model, max_tokens, temperature)

    output_str = completion.choices[0].message.content.strip()
    entities = []

    print(output_str)
    res = re.findall(r"\|.*?\|.*?\|.*?\|.*?\|", output_str)
    count = 0
    for so in res:
        count += 1
        if count <= 2:
            continue
        so = so[1:-1].split("|")

        if len(so) == 4:
            s, o, on, off = so
            try:
                entities.append({'entity': o, 'type': entity_type_dict[s], 'start_idx': int(on), 'end_idx': int(off)})
            except:
                continue

    print(entities)

    return entities


def get_predictions(examples, num_sentences, num_examples):
    predictions = []
    texts = ''
    for i, example in enumerate(examples):
        texts += '文本：\n' + example.text + '\n'
        if ((i + 1) % num_sentences == 0):
            try:
                entities = predict_ner(openai_key=openai_key, sentences=texts, num_sentence=num_sentences,
                                       examples=get_examples(num_examples))
            except openai.error.InvalidRequestError:
                time.sleep(20)
                entities = predict_ner(openai_key=openai_key, sentences=texts, num_sentence=num_sentences,
                                       examples=get_examples(num_examples - 1))

            predictions.append({"text": texts[4:-1], "entities": entities})
            # print(texts)
            # print(entities)
            texts = ''

    return predictions


def generate_test_results(num_examples, num_sentences, cblue_root='../data/CBLUEDatasets'):
    test_examples = EEDataloader(cblue_root, augment=False).get_data("select_dev")[301:500]
    output_dir = "../ckpts/chatgpt_api"
    exist_answer = json.load(open(join(output_dir, "CMeEE_dev_select.json"), "r", encoding="utf8"))
    final_answer = get_predictions(test_examples, num_sentences=num_sentences, num_examples=num_examples)
    final_answer = exist_answer + final_answer
    with open(join(output_dir, "CMeEE_dev_select.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)


def calculate_f1_score(ground_truth_file, predicted_file):
    # Load the ground truth and prediction data from the JSON files
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.loads(f.read())
    with open(predicted_file, 'r') as f:
        predicted_data = json.loads(f.read())
    print(ground_truth_data.__len__(), predicted_data.__len__())

    TP = 0  # True Positives
    ground_length = 0
    predicted_length = 0

    for i in range(len(ground_truth_data)):
        if i > len(predicted_data) - 1:
            break

        ground_truth_entities = ground_truth_data[i]['entities']

        predicted_entities = predicted_data[i]['entities']

        ground_length += sum([len(ground_entity) for ground_entity in ground_truth_entities])
        predicted_length += sum([len(predicted_entity) for predicted_entity in predicted_entities])

        for predicted_entity in predicted_entities:
            for ground_entity in ground_truth_entities:
                if predicted_entity['entity'] == ground_entity['entity'] and \
                        predicted_entity['type'] == ground_entity['type'] and \
                        predicted_entity['start_idx'] == ground_entity['start_idx'] and \
                        predicted_entity['end_idx'] == ground_entity['end_idx']:
                    TP += 1
                    break

    f1 = 2 * TP / (ground_length + predicted_length)

    return f1


def main(_args: List[str] = None):
    # ===== Parse arguments =====

    # generate_test_results(num_examples=3, num_sentences=1)
    print(calculate_f1_score(predicted_file='../ckpts/chatgpt_api/predicted_dev_select.json',
                             ground_truth_file='../data/CBLUEDatasets/CMeEE/select_dev.json'))

    # ===== Set random seed =====


if __name__ == '__main__':
    main()
