#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, QuestionAnswerPrompt

'''
llama_index增强检索
vectorIndex是一种利用向量搜索技术来检索相关资料的系统。
它使用了OpenAI的embedding接口，这个接口可以将文本转换为向量，也就是一组数值，表示文本的语义信息。
当用户输入一个问题时，vectorIndex会将问题的向量与它内部存储的向量进行比较，找出最相似的几个向量。
然后，它会将这些向量再转换回文本，作为与问题相关的资料。
最后，它会将问题和相关资料一起发送给GPT，让GPT生成一个合理的答案。
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    # 加载目录中的语料
    documents = SimpleDirectoryReader('./data/mr_fujino').load_data()
    # 构建索引，持久化
    index = GPTSimpleVectorIndex.from_documents(documents)
    index.save_to_disk('index_mr_fujino.json')

    # 读取持久化文件
    index = GPTSimpleVectorIndex.load_from_disk('index_mr_fujino.json')
    # 提问
    response = index.query("鲁迅先生在日本学习医学的老师是谁？")
    print(response)
    response = index.query("鲁迅先生去哪里学的医学？")
    print(response)

    # 只考虑当前只是，回答是否知道问题答案
    query_str = "鲁迅先生去哪里学的医学？"
    DEFAULT_TEXT_QA_PROMPT_TMPL = (
        "Context information is below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the question: {query_str}\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(DEFAULT_TEXT_QA_PROMPT_TMPL)
    response = index.query(query_str, text_qa_template=QA_PROMPT)
    print(response)

    # 只考虑当前知识，无法回答不相关的问题
    QA_PROMPT_TMPL = (
        "下面的“我”指的是鲁迅先生 \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "根据这些信息，请回答问题: {query_str}\n"
        "如果您不知道的话，请回答不知道\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
    response = index.query("请问林黛玉和贾宝玉是什么关系？", text_qa_template=QA_PROMPT)
    print(response)

