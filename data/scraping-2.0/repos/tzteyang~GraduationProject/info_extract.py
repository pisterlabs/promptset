import time
import tiktoken
import jsonlines
import openai
import os
import sys
import json
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
BASE_DIR = str(Path(__file__).resolve().parent)
sys.path.append(BASE_DIR)
PROMPT_FILE = BASE_DIR + '/few_shot_prompt.txt'
RET_FILE = BASE_DIR + '/Data/extract_res_1000_0428.json'

tokenizer = AutoTokenizer.from_pretrained('gpt2')


def get_key():
    return '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'


def openai_query(content, apikey):
    os.environ["http_proxy"] = "127.0.0.1:7890"
    os.environ["https_proxy"] = "127.0.0.1:7890"
    openai.api_key = apikey
    cnt = 0
    while cnt < 10:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # gpt-3.5-turbo-0301
                messages=[
                    {"role": "user", "content": content}
                ],
                temperature=0.15, # 控制生成的随机性 0-1越高随机性越强
                max_tokens=832, # 生成内容的最大token限制
                frequency_penalty=0,
                presence_penalty=0,
            )
            del os.environ["http_proxy"]
            del os.environ["https_proxy"]
            return response.choices[0].message.content
        except Exception as e: 
            cnt += 1
            time.sleep(5)
            print('openai接口请求出错或请求过快:\n',str(e))
    del os.environ["http_proxy"]
    del os.environ["https_proxy"]


def text_to_chunks(text, chunk_size=2000, overlap=100):
    
    token_ids = tokenizer.encode(text, truncation=False)
    # print(token_ids)
    # print(tokenizer.decode(token_ids))
    tokens_count = len(token_ids)
    
    chunks = []
    for i in range(0, tokens_count, chunk_size-overlap):
        chunk = token_ids[i:(i + chunk_size)]
        chunks.append(chunk)
    return chunks


def top_content_clean(info_list):
    extract_res_list = []
    for info in tqdm(info_list):
        if info['filtered_content'] == '': continue
        text = info['filtered_content']
        chunks = text_to_chunks(text)
        cleaned_text = ""

        index = 0
        for chunk in chunks:
            index += 1
            with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
                prompt = f.read()
            prompt = prompt.replace('{{person}}', info['name'])
            prompt_ids = tokenizer.encode(prompt, truncation=False)
            print(len(prompt_ids))
            prompt = prompt.replace('{{text}}', tokenizer.decode(chunk)).replace('<SEP>', ' ')
            print(prompt + '\n' + '=' * 30)
            completion = openai_query(prompt, get_key())
            try:
                completion_json = json.loads(completion)
                completion_json['id'] = info['id']
                completion_json['chunk_index'] = index
                completion_json['url'] = info['url']
                completion_json['name'] = info['name']
                completion_json['institute'] = info['institute']
                completion_json['main_info'] = info['main_info']

                extract_res_list.append(completion_json)

                with open('./Data/extract_res_1000_ner_v1json', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(completion_json, ensure_ascii=False))
                    f.write('\n')
            except Exception as e:
                print('\napi返回格式有误\n'+str(e))
    
    return extract_res_list


def value_check(value):
    # if value == '' or value == '空' or '未在文本中出现' in value:
    if value == 'unk':
        return False
    return True


def hit_rate_calculate():
    extract_datas = []
    with open(RET_FILE, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            extract_datas.append(json.loads(line))
    # 按照姓名分组处理
    
    grouped_datas, group = [], []
    last_name = ''
    for data in extract_datas:
        if last_name != '' and data['name'] != last_name:
            grouped_datas.append(group.copy()) # 注意要添加列表的浅拷贝
            group.clear()
        group.append(data)
        last_name = data['name']
    grouped_datas.append(group)
    # print(grouped_datas[:10])
    # print(len(grouped_datas))
    occupation, edu_background, resume, achievement, main_info = 0, 0, 0, 0, 0
    count = 0
    cred, part_cred, uncred = 0, 0, 0
    for group in grouped_datas:
        o_tag, e_tag, r_tag, a_tag = False, False, False, False
        for data in group:
            o_tag |= value_check(data['当前职位'])
            e_tag |= value_check(data['工作教育履历'])
            if '个人简介' in data:
                r_tag |= value_check(data['个人简介'])
            if '个人简历' in data:
                r_tag |= value_check(data['个人简历'])
            a_tag |= value_check(data['奖项成就'])

        count += 1 if (o_tag or e_tag or r_tag or a_tag) else 0
        occupation += 1 if o_tag else 0
        edu_background += 1 if e_tag else 0
        resume += 1 if r_tag else 0
        achievement += 1 if a_tag else 0
        main_info += 1 if (e_tag or r_tag or a_tag) else 0

    #     if e_tag or r_tag or a_tag:
    #         cred_tag, uncred_tag = 0, 0
    #         for data in group:
    #             o2_tag, e2_tag, r2_tag, a2_tag = False, False, False, False
    #             o2_tag |= value_check(data['当前职位'])
    #             e2_tag |= value_check(data['工作教育履历'])
    #             if '个人简介' in data:
    #                 r2_tag |= value_check(data['个人简介'])
    #             if '个人简历' in data:
    #                 r2_tag |= value_check(data['个人简历'])
    #             a2_tag |= value_check(data['奖项成就'])
    #             if data['cred'] == 1:
    #                 cred_tag += 1
    #             else:
    #                 uncred_tag += 1
            
    #         if cred_tag == len(group):
    #             cred += 1
    #         elif uncred_tag == len(group):
    #             uncred += 1
    #         else:
    #             part_cred += 1
    #         assert(cred_tag + uncred_tag == len(group))
    #         print(group)
    #         print(cred_tag, uncred_tag, len(group))
    #         print('=' * 30)
    # print('gpt抽取流程准确率:')
    # print(f'完全正确: {cred}/{main_info}, 概率: {cred/main_info}')
    # print(f'部分正确: {part_cred}/{main_info}, 概率: {part_cred/main_info}')
    # print(f'不正确: {uncred}/{main_info}, 概率: {uncred/main_info}')

    print('=' * 30)    

    print(f'主要关注字段得到补充的专家数: {main_info}')
    print(f'任职 字段得到补充的专家数: {occupation}')
    print(f'工作教育履历 字段得到补充的专家数: {edu_background}')
    print(f'个人简介 字段得到补充的专家数: {resume}')
    print(f'奖项成就 字段得到补充的专家数: {achievement}')
    """
        v1
        任意字段得到补充的专家数: 47
        任职 字段得到补充的专家数: 36
        毕业院校 字段得到补充的专家数: 16
        个人履历 字段得到补充的专家数: 26
        研究领域 字段得到补充的专家数: 21
        奖项成就 字段得到补充的专家数: 19
        v2
        任意字段得到补充的专家数: 49
        任职 字段得到补充的专家数: 43
        毕业院校 字段得到补充的专家数: 17
        个人履历 字段得到补充的专家数: 21
        研究领域 字段得到补充的专家数: 15
        奖项成就 字段得到补充的专家数: 13
        v3-0428
        主要关注字段得到补充的专家数: 134
        任职 字段得到补充的专家数: 148
        工作教育履历 字段得到补充的专家数: 91
        个人简介 字段得到补充的专家数: 107
        奖项成就 字段得到补充的专家数: 80
    """
    """
        5/47
        科研之友网站
        badcase的原因：
        1. ner工具的准确率问题，漏掉某些人物实体，带进了干扰噪声。
        2. 文本切块时，将关键人物姓名和其个人信息分隔开来，造成了错误的抽取。
    """


if __name__ == '__main__':
    # pass
    # top_content_clean()
    hit_rate_calculate()
    