import json
import os

import openai
import requests

from pd_utils import pd_concat
from file_utils import read_txt_from_file, split_text
from data_utils import question_json_to_dataframe
from llm_chat import ask_llm


def directory_files_to_tag_and_question(directory_file, document):
    df = None
    for filename in os.listdir(directory_file):
        if filename == '.DS_Store':
            continue
        file_path = os.path.join(directory_file, filename)
        print(file_path)
        df1 = extract_question(file_path, title=filename, url=None, filename=filename)
        if df1 is None:
            continue
        if df is None:
            df = df1
        else:
            df = pd_concat([df, df1])

    df.to_excel("%s.xlsx" % document, index=False)
    return "%s.xlsx"


def extract_question(file_path, title=None, url=None):
    content = read_txt_from_file(file_path)
    length = len(content)
    if length == 0:
        return None
    if length > 1000:
        contents = split_text(content, 1000)
    else:
        contents = [content]
    df_list = []
    for chunk in contents:
        s_prompt = f'''下面的文本来自于xx，主题是{title}，请仔细分析，告诉我这个文本可以回答哪些问题'''
        df = extract_question_from_content(chunk, s_prompt=s_prompt)
        if df is not None:
            df["原文"] = chunk
            df_list.append(df)
    df = pd_concat(df_list)
    if title:
        file_path = title
    df["来源"] = file_path
    if url:
        df["原文链接"] = url

    return df


def extract_question_from_content(content, retries=3, s_prompt=None):
    length = len(content)
    c1 = round(length / 1000)
    c2 = round(length / 800)
    final_prompt = '你是一位善于读书的学者。下面这篇这段文本来自书本扫描件，有比较多的ocr识别错误，也有不必要的页眉页脚干扰。请忽略所有的干扰项，通读文档，告诉我这篇文档可以回答哪些问题。'
    if s_prompt:
        final_prompt = s_prompt
    print(content)
    prompt = f'''{final_prompt}请提供至少{c1}个主要问题，和{c2}个次要问题。

……

--- 

{content}


---
注意：
1.xx
2.请以json 格式给出可能的问题,问题里不要带“这段对话”““这个文本”“这个”等信息，问题要尽可能的覆盖文本的各个部分
3.yy

{{
"主要回答的问题":[],
"还能回答的问题":[]
}}
'''
    for retry_count in range(retries):
        try:
            result = json.loads(ask_llm(prompt, channel='Chato'))
            print(result)
            df = question_json_to_dataframe(result)
            return df
        except:
            if retry_count == retries - 1:
                print("An error occurred and all retries failed.")
                return None
            else:
                print(f"An error has occurred. Retrying in 5 seconds. ({retries - retry_count - 1} attempts left)")
    return None


if __name__ == '__main__':
    base_directory = '/Users/baixing/Desktop/'
    document = 'sample_folder'
    directory_file = base_directory + document
    directory_files_to_tag_and_question(directory_file, document)