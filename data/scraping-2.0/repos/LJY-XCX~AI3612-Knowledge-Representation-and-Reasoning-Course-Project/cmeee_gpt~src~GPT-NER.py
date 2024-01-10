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
import threading 
import torch
import numpy as np

openai_key = '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'
openai.api_base = "https://chatie.deno.dev/v1"

entity_type_dict = {'疾病': 'dis', '临床表现': 'sym', '医疗程序': 'pro', '医疗设备': 'equ', '药物': 'dru', '医学检验项目': 'ite',
                    '身体': 'bod', '科室': 'dep', '微生物类': 'mic'}

reverse_entity_type_dict = {'dis': '疾病', 'sym': '临床表现', 'pro': '医疗程序', 'equ': '医疗设备', 'dru': '药物', 'ite': '医学检验项目',
                            'bod': '身体', 'dep':'科室', 'mic':'微生物类'}
num_gt = 0
num_prediction = 0
num_intersection = 0
def get_squeezed_embedding():
    if (os.path.exists('B.npy')):
        return np.load('B.npy')
    embedding = np.zeros(shape=(len(data_train), 500))
    for i, data in enumerate(data_train):
        logits = bert(data).detach().numpy()
        embedding[i,:] = np.dot(trans_matrix, logits.transpose()).transpose()
    np.save('B.npy', embedding)
    return embedding

def get_test_embedding():
    if (os.path.exists('C.npy')):
        return np.load('C.npy')
    test_embedding = np.zeros(shape=(len(data_test), 500))
    for i, data in enumerate(data_test):
        logits = bert(data).detach().numpy()
        test_embedding[i, :] = np.dot(trans_matrix, logits.transpose()).transpose()
    np.save('C.npy', test_embedding)
    return test_embedding

def find_substring_indices(string, substring):
    start_index = string.find(substring)
    if start_index == -1:
        return None, None  # Substring not found
    end_index = start_index + len(substring) - 1
    return start_index, end_index

def get_completion(prompt, model='gpt-3.5-turbo', max_tokens=1024, temperature=0.5):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stop=None,
        n=1,
    )
    return completion


def get_examples(num_examples, cblue_root='../data/CBLUEDatasets'):

    example_list = []
    for i in range(num_examples):
        
        examples = EEDataloader(cblue_root).get_data("train") 

        idx = random.randint(0, len(examples))
        text = examples[idx].text
        entity_list = examples[idx].entities
        if False:
            offset = 0
            end = -10
            for item in entity_list:
                if item['start_idx'] <= end:
                    continue
                index = item['start_idx'] + offset
                text = text[:index] + "#" + text[index:]
                offset += 1
                index = item['end_idx'] + offset + 1
                text = text[:index] + "#" + text[index:]
                offset += 1
                end = index - 1
        prompt_text = f"文本：{text}\n"

        for _, entity_dict in enumerate(entity_list):
            entity_type = reverse_entity_type_dict[entity_dict['type']]
            entity = entity_dict['entity']
            prompt_entity = f"实体类别-{entity_type}; 实体-{entity}\n"
            prompt_text += prompt_entity
    
        example_list.append(prompt_text)

    return example_list

def get_examples_with_specific_type(num_examples, cblue_root='../data/CBLUEDatasets', specific_type="dis"):

    example_list = []
    example_num = 0
    while example_num < num_examples:
        
        examples = EEDataloader(cblue_root).get_data("train") 

        idx = random.randint(0, len(examples))
        text = examples[idx].text
        entity_list = examples[idx].entities
        if False:
            offset = 0
            end = -10
            for item in entity_list:
                if item['start_idx'] <= end:
                    continue
                index = item['start_idx'] + offset
                text = text[:index] + "#" + text[index:]
                offset += 1
                index = item['end_idx'] + offset + 1
                text = text[:index] + "#" + text[index:]
                offset += 1
                end = index - 1
        prompt_text = f"文本：{text}\n"
        num = 0
        for _, entity_dict in enumerate(entity_list):
            if entity_dict['type'] == specific_type:
                entity_type = reverse_entity_type_dict[entity_dict['type']]
                entity = entity_dict['entity']
                prompt_entity = f"实体类别-{entity_type}; 实体-{entity}\n"
                prompt_text += prompt_entity
                num += 1
        if num>0:
            example_list.append(prompt_text)
            example_num += 1 

    return example_list

def get_examples_using_similarity(num_examples, cblue_root='../data/CBLUEDatasets', test_example_index=0):
    example_list = []
    examples = EEDataloader(cblue_root).get_data("train") 
    embeddings = get_squeezed_embedding()
    test_embedding = get_test_embedding()
    num_train_data = len(examples)
    dis = np.zeros(shape=(num_train_data))
    for j in range(num_train_data):
        dis[j] = np.linalg.norm(embeddings[j,:]-test_embedding[test_example_index,:])
    num, indexes = torch.from_numpy(dis).topk(5, largest=False)
    for idx in indexes:
        text = examples[idx].text
        entity_list = examples[idx].entities
        prompt_text = f"文本：{text}\n"
        for _, entity_dict in enumerate(entity_list):
            entity_type = reverse_entity_type_dict[entity_dict['type']]
            entity = entity_dict['entity']
            prompt_entity = f"实体类别-{entity_type}; 实体-{entity}\n"
            prompt_text += prompt_entity

        example_list.append(prompt_text)
    return example_list

    

def predict_ner(openai_key, sentences, examples, num_sentence, model='gpt-3.5-turbo', max_tokens=1024, temperature=0.5,
                ent_type_dic=entity_type_dict):
    openai.api_key = openai_key
    use_reflection = False
    output_dic = {}
    examples_str = "\n".join(examples)
    prompt = (f"你是一个医疗领域的专家。请你从给出的句子中提取出实体和对应的实体类别。你在分辨实体要仔细考虑实体的起始和终止位置，选出最有可能的实体。你要仔细思考实体类别，在不同实体类型中选出最有可能的。\n"
              f"注意：你在预测实体类别时只能从以下九种类别中选取：1.疾病、2.临床表现、3.医疗程序、4.医疗设备、5.药物、6.医学检验项目、7.身体、8.科室、9.微生物类。\n"
              f"输入文本和输出实体的格式如下所示：\n{examples_str}\n"
              f"以下是你要进行命名实体识别的输入文本：{sentences}\n"
              )
    try:
        completion = get_completion(prompt, model, max_tokens, temperature)
    except openai.error.RateLimitError:
        print("yes")
        time.sleep(20)
        completion = get_completion(prompt, model, max_tokens, temperature)

    output_str = completion.choices[0].message.content.strip()
    if use_reflection:
        prompt_reflection = (f"你是一个医疗领域的专家。请你判断以下实体的实体类别是否正确，并且改正，以相同的格式输出\n。"
                             f"注意：你在预测实体类别时只能从以下九种类别中选取：1.疾病、2.临床表现、3.医疗程序、4.医疗设备、5.药物、6.医学检验项目、7.身体、8.科室、9.微生物类。\n"
                             f"以下是需要检查的实体识别结果：{output_str}\n"
                  )
        prompt = prompt_reflection
        try:
            completion = get_completion(prompt, model, max_tokens, temperature)
        except openai.error.RateLimitError:
            time.sleep(20)
            completion = get_completion(prompt, model, max_tokens, temperature)
        output_str = completion.choices[0].message.content.strip()
    output = []
    for item in output_str.split('\n'):
        if "实体类别-" not in item or "实体-" not in item:
            continue
        if item.split('实体类别-')[1].split(';')[0] not in ent_type_dic.keys():
            continue
        output_dic = {}
        item = item.strip()
        start_idx, end_idx =find_substring_indices(sentences, item.split('实体-')[1].split(';')[0])
        if start_idx == None:
            continue
        output_dic['start_idx'] = start_idx - 3
        output_dic['end_idx'] = end_idx - 3   
        output_dic['type'] = ent_type_dic[item.split('实体类别-')[1].split(';')[0]]
        output_dic['entity'] = item.split('实体-')[1].split(';')[0]    
        output.append(output_dic)

    return output

def predict_ner_pair_choice(openai_key, sentences, examples, num_sentence, model='gpt-3.5-turbo', max_tokens=1024, temperature=0.5,
                ent_type_dic=entity_type_dict, specific_type="dis"):
    openai.api_key = openai_key
    use_reflection = False
    output_dic = {}
    examples_str = "\n".join(examples)
    prompt = (f"你是一个医疗领域的专家。请你从给出的句子中提取出所有的{reverse_entity_type_dict[specific_type]}实体。你在分辨实体要仔细考虑实体的起始和终止位置，选出最有可能的实体。\n"
              f"输入文本和输出实体的格式如下所示：\n{examples_str}\n"
              f"以下是你要进行命名实体识别的输入文本：{sentences}\n"
              )
    try:
        completion = get_completion(prompt, model, max_tokens, temperature)
    except openai.error.RateLimitError:
        time.sleep(20)
        completion = get_completion(prompt, model, max_tokens, temperature)

    output_str = completion.choices[0].message.content.strip()
    output = []
    for item in output_str.split('\n'):
        if "实体类别-" not in item or "实体-" not in item:
            continue
        if item.split('实体类别-')[1].split(';')[0] not in ent_type_dic.keys():
            continue
        output_dic = {}
        item = item.strip()
        start_idx, end_idx =find_substring_indices(sentences, item.split('实体-')[1].split(';')[0])
        if start_idx == None:
            continue
        output_dic['start_idx'] = start_idx - 3
        output_dic['end_idx'] = end_idx - 3   
        output_dic['type'] = ent_type_dic[item.split('实体类别-')[1].split(';')[0]]
        output_dic['entity'] = item.split('实体-')[1].split(';')[0]    
        output.append(output_dic)

    return output
def thread_process(example, num_sentences, num_examples, index):
    multi_choice = True
    similarity = False
    lock = threading.Lock()
    global num_gt
    global num_prediction
    global num_intersection
    texts = ''
    texts += '文本：' + example.text + '\n'
    with lock:
        num_gt += len(example.entities)
    if multi_choice:
        if similarity:
            entities = predict_ner(openai_key=openai_key, sentences=texts, num_sentence=num_sentences, examples=get_examples_using_similarity(num_examples, test_example_index=index))         
        else:
            entities = predict_ner(openai_key=openai_key, sentences=texts, num_sentence=num_sentences, examples=get_examples(num_examples))
    else:
        total_entities = []
        for entity_type in reverse_entity_type_dict.keys():
            entities = predict_ner_pair_choice(openai_key=openai_key, sentences=texts, num_sentence=num_sentences, examples=get_examples_with_specific_type(num_examples, specific_type=entity_type), specific_type=entity_type)
            total_entities.extend(entities)
    with lock:
        num_prediction += len(entities)
    for entity in entities:
        if entity in example.entities:
            with lock:
                num_intersection += 1
def get_predictions(examples, num_sentences, num_examples):
    predictions = []
    threads = []
    for i, example in enumerate(examples):
        thread = threading.Thread(target=thread_process, args=(example,num_sentences, num_examples, i))
        threads.append(thread)
        thread.start()
        if i%100 ==0 and i > 0:
            for thread in threads:
                thread.join()
            threads = []

    print("F1 score is:"+str(2*num_intersection/(num_gt + num_prediction)))
    
    return predictions

def generate_test_results(num_examples, num_sentences, cblue_root='../data/CBLUEDatasets'):

    test_examples = EEDataloader(cblue_root).get_data("select")
    output_dir = "../ckpts/chatgpt_api"
    final_answer = get_predictions(test_examples, num_sentences=num_sentences, num_examples=num_examples)
    with open(join(output_dir, "CMeEE_test.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)


def main(_args: List[str] = None):
    # ===== Parse arguments =====
    
    generate_test_results(num_examples=5, num_sentences=1)
    
    # ===== Set random seed =====

    
if __name__ == '__main__':
    main()
