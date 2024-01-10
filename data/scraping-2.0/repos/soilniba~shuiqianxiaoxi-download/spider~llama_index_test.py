from bs4 import BeautifulSoup
import urllib
import json
import time
import datetime
import requests
import os
import re
import gzip
import PyPDF2
import docx2txt
import nltk
import html2text
import openai
from loguru import logger
import logging
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index import (
    GPTKeywordTableIndex,
    GPTSimpleVectorIndex,
    SimpleDirectoryReader,
    BeautifulSoupWebReader,
    StringIterableReader,
    LLMPredictor,
    PromptHelper,
    QuestionAnswerPrompt,
    RefinePrompt,
    ServiceContext
)
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL, DEFAULT_REFINE_PROMPT_TMPL
from config import openai_api_key, feishu_robot_news, feishu_robot_error

script_dir = os.path.dirname(os.path.realpath(__file__))    # 获取脚本所在目录的路径
os.chdir(script_dir)                                        # 切换工作目录到脚本所在目录
openai.api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
import psutil
p = psutil.Process()  # 获取当前进程的Process对象
p.nice(psutil.IDLE_PRIORITY_CLASS)  # 设置进程为低优先级

feishu_robot_news = feishu_robot_error  # 强制使用测试频道

def get_article(href):
    response = requests.get(href)
    html = response.content
    # 解析网页内容
    soup = BeautifulSoup(html, 'html.parser')
    # 提取网页正文
    text = soup.get_text()
    # 去除多余空格、换行符等无用字符
    text = re.sub(r'\s+', ' ', text).strip()
    # 将多个连续空格替换为一个空格
    text = re.sub(r'\s+', ' ', text)
    # 输出处理后的文本
    # print(url, text)
    return text

def ask_llama_index(href = None, text = None, json_filename = None):
    # define LLM
    # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=2048))
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=2048))

    # define prompt helper
    # set maximum input size
    max_input_size = 2048
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    chunk_size_limit = 10000
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    save_json_path = json_filename and f'{os.path.dirname(__file__)}\\{json_filename}'
    if not save_json_path or not os.path.isfile(save_json_path):
        # doc是你文档所存放的位置，recursive代表递归获取里面所有文档
        # documents = SimpleDirectoryReader(input_dir=os.path.dirname(__file__) + '/doc',recursive=True).load_data()
        if href:
            documents = BeautifulSoupWebReader().load_data([href])
        if text:
            documents = StringIterableReader().load_data(texts=[text])
        for doc in documents:
            doc.text = doc.text.replace("。", ". ")
        # index = GPTSimpleVectorIndex.from_documents(documents)
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
        # index = GPTSimpleVectorIndex.from_documents(documents)
        if save_json_path:
            index.save_to_disk(save_json_path)
    else:
        index = GPTSimpleVectorIndex.load_from_disk(save_json_path, service_context=service_context)


    # Context information is below. 
    # ---------------------
    # {context_str}
    # ---------------------
    # Given the context information and not prior knowledge, answer the question: {query_str}
    text_qa_prompt_tmpl = (
        "我们在下面提供了上下文信息. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "鉴于此信息, 请回答以下问题: {query_str}\n"
    )
    # The original question is as follows: {query_str}
    # We have provided an existing answer: {existing_answer}
    # We have the opportunity to refine the existing answer (only if needed) with some more context below.
    # ------------
    # {context_msg}
    # ------------
    # Given the new context, refine the original answer to better answer the question. If the context isn't useful, return the original answer.
    refine_prompt_tmpl = (
        "之前我们询问过这个问题: {query_str}\n"
        "得到了这样一个答案: {existing_answer}\n"
        "现在我们有机会完善现有的答案 (仅在需要时) 通过下面的更多上下文.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "给我一个新的答案, 完善原始答案以更好的回答问题. 如果新的上下文没有用或者没必要再完善了, 则重复一遍原始的答案.\n"
    )
    text_qa_prompt = QuestionAnswerPrompt(text_qa_prompt_tmpl)
    refine_prompt = RefinePrompt(refine_prompt_tmpl)


    # answer = index.query("请尽可能详细的总结文章概要,并使用换行使阅读段落更清晰", 
    #                      text_qa_template = text_qa_prompt,
    #                      refine_template = refine_prompt)
    # print(answer)
    while True:
        ask = input("请输入你的问题：")
        print(index.query(ask, 
                         text_qa_template = text_qa_prompt,
                         refine_template = refine_prompt))
    return answer.response


def read_tab(file_path):
    texts = []  # 创建一个空列表，用于存储文件内容

    with open(file_path, 'r', encoding='utf-8') as f:  # 打开文件并读取内容
        lines = f.readlines()  # 逐行读取文件内容并存储在一个列表中
        count = 0  # 创建一个计数器，用于跳过前三行
        for line in lines:
            if count >= 3:  # 如果计数器大于等于3，说明已经跳过了前三行，可以将该行文本内容添加到texts列表中
                texts.append(line.strip())  # 去掉每行末尾的换行符并将其添加到texts列表中
            else:
                count += 1  # 如果计数器小于3，说明仍需要跳过该行，将计数器加1
    return texts

def create_vector_index_help_guide():
    # define LLM
    # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=2048))
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=2048))

    # define prompt helper
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 2560
    # set maximum chunk overlap
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # doc是你文档所存放的位置，recursive代表递归获取里面所有文档
    # documents = SimpleDirectoryReader(input_dir=os.path.dirname(__file__) + '/doc',recursive=True).load_data()
    # documents = BeautifulSoupWebReader().load_data([url])
    texts = read_tab('E:\\game\\pub\\data\\tables\\player\\helper_guide.tab')
    documents = StringIterableReader().load_data(texts=texts)
    # index = GPTSimpleVectorIndex.from_documents(documents)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    # index = GPTSimpleVectorIndex.from_documents(documents)
    save_json_path = os.path.dirname(__file__) + '\\helper_guide.json'
    index.save_to_disk(save_json_path);

def ask_by_helper_guide():
    # define LLM
    # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=2048))
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=2048))

    # define prompt helper
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    chunk_size_limit = 10000
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # query_index.py   从index文件里获得相关资料并向GPT提问
    save_json_path = os.path.dirname(__file__) + '\\helper_guide.json'
    index = GPTSimpleVectorIndex.load_from_disk(save_json_path, service_context=service_context)

    # Context information is below. 
    # ---------------------
    # {context_str}
    # ---------------------
    # Given the context information and not prior knowledge, answer the question: {query_str}
    text_qa_prompt_tmpl = (
        "我们在下面提供了上下文信息. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "鉴于此信息，请回答以下问题: {query_str}\n"
    )
    # The original question is as follows: {query_str}
    # We have provided an existing answer: {existing_answer}
    # We have the opportunity to refine the existing answer (only if needed) with some more context below.
    # ------------
    # {context_msg}
    # ------------
    # Given the new context, refine the original answer to better answer the question. If the context isn't useful, return the original answer.
    refine_prompt_tmpl = (
        "之前我们询问过这个问题: {query_str}\n"
        "得到了这样一个答案: {existing_answer}\n"
        "现在我们有机会完善现有的答案 (仅在需要时) 通过下面的更多上下文.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "给我一个新的答案, 完善原始答案以更好的回答问题. 如果新的上下文没有用, 则返回原始的答案.\n"
    )
    text_qa_prompt = QuestionAnswerPrompt(text_qa_prompt_tmpl)
    refine_prompt = RefinePrompt(refine_prompt_tmpl)

    while True:
        ask = input("请输入你的问题：")
        print(index.query(ask, 
                         text_qa_template = text_qa_prompt,
                         refine_template = refine_prompt))

# create_vector_index_help_guide()
# logging.getLogger('llama_index.token_counter.token_counter').setLevel(logging.WARNING)
# ask_by_helper_guide()

# ask_llama_index('https://mp.weixin.qq.com/s/wY-DkYOaar1Z3Hy4eBPebg', None, 'wY-DkYOaar1Z3Hy4eBPebg.json')
ask_llama_index('https://zhuanlan.zhihu.com/p/623585339')

# 从doc/pormpt_tags.txt文件读入text信息
# text = open('doc\\pormpt_tags.txt', 'r').read()
# ask_llama_index(None, text, 'pormpt_tags.json')
