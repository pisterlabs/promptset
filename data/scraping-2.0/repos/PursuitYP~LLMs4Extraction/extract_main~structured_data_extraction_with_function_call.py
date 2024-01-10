""" LLMs for DeepShovel: 结构化数据抽取 """
import os
import openai
import textract
import tiktoken
from dotenv import load_dotenv
import os
import regex as re
from tqdm import tqdm
import pandas as pd
import csv
import random
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader
import ast
import json
from pydantic import Field, BaseModel
from openai_function_call import OpenAISchema


# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


# 使用gpt-3.5-turbo抽取数据，加入异常处理机制
def extract_chunk(document, template_prompt):
    for i in range(3):  # Retry the API call up to 3 times
        try:
            prompt=template_prompt.replace('<document>', document)
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo', 
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return "1. " + response['choices'][0]['message']['content']
        except openai.error.RateLimitError:  # If rate limit is exceeded
            wait_time = (2 ** i) + random.random()  # Exponential backoff with jitter
            logging.warning(f"Rate limit exceeded. Retrying after {wait_time} seconds.")
            time.sleep(wait_time)  # Wait before retrying
        except Exception as e:  # If any other error occurs
            logging.error(f"API call failed: {str(e)}")
            return None  # Return None for failure
    logging.error("Failed to call OpenAI API after multiple retries due to rate limiting.")
    return None  # Return None for failure


# 调用API并使用重试机制处理rate limit error和其他异常
def get_completion(prompt, model="gpt-3.5-turbo"):
    for i in range(3):  # Retry the API call up to 3 times
        try:
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message["content"]
        except openai.error.RateLimitError:  # If rate limit is exceeded
            wait_time = (2 ** i) + random.random()  # Exponential backoff with jitter
            logging.warning(f"Rate limit exceeded. Retrying after {wait_time} seconds.")
            time.sleep(wait_time)  # Wait before retrying
        except Exception as e:  # If any other error occurs
            logging.error(f"API call failed: {str(e)}")
            return None  # Return None for failure
    logging.error("Failed to call OpenAI API after multiple retries due to rate limiting.")
    return None  # Return None for failure

# 调用API并使用重试机制处理rate limit error和其他异常，调用function call功能
def get_completion_function_call(prompt, attribute_set_string):
    class AttributeDict(OpenAISchema):
        """Attributes of user input"""
        exec(attribute_set_string)

    for i in range(3):  # Retry the API call up to 3 times
        try:
            messages = [{"role": "system", "content": "Use AttributeDict to parse this data."}, 
                        {"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                functions = [AttributeDict.openai_schema],
                messages=messages,
                temperature=0,
            )
            return response
        except openai.error.RateLimitError:  # If rate limit is exceeded
            wait_time = (2 ** i) + random.random()  # Exponential backoff with jitter
            logging.warning(f"Rate limit exceeded. Retrying after {wait_time} seconds.")
            time.sleep(wait_time)  # Wait before retrying
        except Exception as e:  # If any other error occurs
            logging.error(f"API call failed: {str(e)}")
            return None  # Return None for failure
    logging.error("Failed to call OpenAI API after multiple retries due to rate limiting.")
    return None  # Return None for failure


# 处理过程信息写入log文件
def log_to_file(log_file, message):
    try:
        with open(log_file, 'a') as file:
            file.write(message + '\n')
    except Exception as e:
        logging.error(f'Failed to log to file {log_file}: {str(e)}')
        raise


# 使用PdfReader读取pdf文献，手动加入Page Number信息
def read_pdf(filepath):
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a pdf reader object
    reader = PdfReader(filepath)
    pdf_text = ""
    page_number = 0
    for page in reader.pages:
        page_number += 1
        pdf_text += page.extract_text() + f"\nPage Number: {page_number}"
    return pdf_text


# 传入pdf路径和带抽取的属性列表，返回抽取的结构化数据
def data_extraction(pdf_path, field_list):
    # 1. 执行pdf解析和切片
    pdf_text = read_pdf(pdf_path)
    clean_text = pdf_text.replace("  ", " ").replace("\n", "; ").replace(';',' ')
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = create_chunks(clean_text, 1000, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]

    # 2. 关键信息抽取：多线程对text_chunks处理，抽取关键信息
    question_format = "0. What is the value of the 'title' attribute"
    # 适应性地生成抽取问题，并集成到抽取提示extract_prompt中去
    for idx, field in enumerate(field_list):
        new_question = str(idx+1) + ". What is the value of the '" + field + "' attribute"
        question_format = question_format + "\n" + new_question
    document = '<document>'
    # 关键信息抽取prompt
    extract_prompt=f'''Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output \"Not specified\".
When you extract a key piece of information, include the closest page number.
---
Use the following format:
{question_format}
---
Document: \"\"\"{document}\"\"\"\n
0. What is the value of the 'title' attribute: Origin of Lower Carboniferous cherts in southern Guizhou, South China (Page 1)
1.'''
    # 多线程对text_chunks处理，抽取关键信息
    results = []
    log_file = 'log_geo_extract.txt'
    # log_to_file(log_file, f'Number of chunks: {len(text_chunks)}')
    with ThreadPoolExecutor() as executor:
        # 多线程处理
        futures = {executor.submit(extract_chunk, chunk, extract_prompt): chunk for chunk in text_chunks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing chunks'):
            # 收集完成的线程处理好的结果
            response = future.result()
            if response is None:
                # log_to_file(log_file, f'Failed to process chunk {futures[future]}')
                pass
            else:
                # 汇总关键信息抽取的结果
                results.append(response)
                # log_to_file(log_file, f'Successfully processed chunk!')
    # 进一步整理关键信息抽取结果，便于下一步格式化转换
    groups = [r.split('\n') for r in results]
    groups = [y for x in groups for y in x]
    groups = sorted(groups)
    groups = [x for x in groups if "Not specified" not in x and "__" not in x]
    zipped = groups
    # 移除太长的结果 (保留len(r) <= 180)
    zipped = [r for r in zipped if len(r) <= 180]

    # 3. 数据格式转换：根据抽取的关键信息，转换生成JSON样式的结果
    zipped_str = str(zipped)[1:][:-1]
    # 数据格式转换prompt
    transform_prompt = f'''I'm going to ask for attributes. 
Use AttributeDict to parse this PARAGRAPH.
Please give your answer based on all the content of the entire PARAGRAPH.
---
PARAGRAPH
{zipped_str}
'''

    # 适应性地生成attribute_set_string, 在AttributeDict类中exec()生成可执行代码
    attribute_set_string = ""
    for idx, field in enumerate(field_list):
        new_attribute = f"""{field.replace(" ", "_")}: str = Field(..., description="{field}")"""
        attribute_set_string = attribute_set_string + "\n" + new_attribute

    class AttributeDict(OpenAISchema):
        """Attributes of user input"""
        exec(attribute_set_string)

    response = get_completion_function_call(transform_prompt, attribute_set_string)
    arrtibutes = AttributeDict.from_response(response)
    res_json = dict(arrtibutes)

    return res_json


if __name__ == '__main__':
    # 环境初始化，用户上传OpenAI API key
    load_dotenv()
    os.environ["http_proxy"] = "http://10.10.1.3:10000"
    os.environ["https_proxy"] = "http://10.10.1.3:10000"
    # os.environ["http_proxy"] = "http://10.10.10.10:17890"
    # os.environ["https_proxy"] = "http://10.10.10.10:17890"
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ['OPENAI_API_KEY'] = openai.api_key

    # 用户上传pdf，输入需要抽取的属性列表
    pdf_path = "data/radiolarian/715.pdf"
    field_list = ["section name", "location of the samples and sections", "GPS location", 
                  "associated fossils", "lithology", "number of species and genera found"]

    # LLMs结构化数据抽取
    res_json = data_extraction(pdf_path, field_list)
    with open('results/result.json', 'w', newline='\n') as file:
        json.dump(res_json, file, indent=4)

    print(json.dumps(res_json, indent=4))
