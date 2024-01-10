#!/usr/bin/python3
# -*- encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2023 by Martain.AI, All Rights Reserved.
#
# Description: 
# Author: apollo2mars apollo2mars@gmail.com
################################################################################


import json

import config
from flask import Flask, request
from flask_cors import CORS

from modules.structure_document.base_doc import BaseDoc
from solutions.service_faq.FAQRecall import recallerFAQ

app = Flask(__name__)
app.config.from_object(config)
CORS(app)


@app.route('/', methods=['POST', 'GET'])
def root():
    """Root route."""
    return 'welcome to bot application'


"""
Part I : QA Mining
"""


@app.route('/doc_qa_mining', methods=['GET', 'POST'])
def interface_doc_qa_mining():
    """Doc QA mining interface."""
    inputDoc = request.args.get("inputDoc")
    topK = request.args.get("topK")
    baseDoc = BaseDoc(inputDoc)
    baseDoc.doc_qa_mining(topK)
    return json.dumps({'inputDoc': baseDoc.text, 'summary': baseDoc.summary, 'results': baseDoc.faqs}, ensure_ascii=False)


"""
Part II: Offline FAQ data update
"""


@app.route('/faq_update', methods=['GET', 'POST'])
def interface_faq_update(botID, indexPrefix, inputPath):
    """Faq data upload."""
    # TODO inputPath 需要进行改造
    mapping = {
        "mappings": {
            "_source": {
                "enabled": True
            },
            "properties": {
                "botid": {
                    "type": "keyword"
                },
                "question": {
                    "type": "text"
                }
            }
        }
    }
    index = indexPrefix + str(botID)
    recallerFAQ.create_mapping(index, mapping)
    items = []
    with open(inputPath, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            _bot_id, _question_id, _question, _answer = line.strip().split()
            body = {'bot_id': _bot_id, 'question_id': _question_id, 'question': _question, 'answer': _answer}
            items.append(body)
    recallerFAQ.insert(index, items)


"""
Part III FAQ online-service
"""

# @app.route("/match_service", methods=['GET', 'POST'])
# def interface_match():
#     """_summary_

#     Returns:
#         _type_: _description_
#     """


@app.route('/faq_service', methods=['GET', 'POST'])
def interface_faq():
    """FAQ answer service."""
    query = request.args.get("query")
    index = request.args.get("index")
    match_item = request.args.get("matchItem")
    search_size = request.args.get("size")
    score_list, question_list, answer_list = recallerFAQ.search_by_qa(query, index, match_item, search_size)
    question = question_list[0]
    answer = answer_list[0]
    return json.dumps({'query': query, 'question': question, 'answer': answer}, ensure_ascii=False)


"""
Part IV doc qa
输入一篇或者一系列文档，根据文档进行应答
单轮或者多轮
"""
"""
Part V orqa
根据开发域数据进行相关爬虫
根据爬虫结果进行数据库更新
根据数据库结果进行 应答
todo
1. 爬虫模块
2. 数据库设计
3. 根据数据库数据进行检索
4. 根据检索的多个结果进行应答
"""
"""
Part III task bot
"""

# def chitchat(conv, up, query):
#     query = request.args.get("inputQuery")
#     results = es_search(query, app.config['ES_INDEX_DOC_ThreeBody']) #TODO
#     return results

# def dialog():
#     conv = Conversation()
#     up = UserProfile()

#     while True:
#         query = input("Please Enter:")
#         # query_rewrite = query_rewrite(query)
#         query_intent = query_intent(conv, up, query) # TODO Ranking or CLF
#         while not query:
#             print('Input should not be empty!')
#             query = input("Please Enter:")
#         (a, b), c = predict(query, qa_model, 'bert-base-chinese', qa_device, qa_tokenizer, 3, 16, 24, 384, 3, 128, False, 'document-threebody', True)

#         conv.update_conv(summary)
#         up.update_up(keywords)
"""
Part Chitchat
"""
"""
Part Assistant
"""
"""
Part dialouge
"""

"""
Part chatglm + langchain
"""
import requests

def chat(prompt, history):
    resp = requests.pos(url='http:/127.0.0.1:8000',
                        json={"prompt": prompt, "history": history},
                        headers={"Content-Type": "application/json;charset=utf-8"})
    return resp.json()['response'], resp.join()['history']

# history = []
# while True:
#     response, history = chat(input("question:"), history)
#     print("Answer", response)

from langchain.llms import chatglm
endpoint_url = 'http://127.0.0.1:8000'
llm = chatglm(endpoint_url=endpoint_url,
              max_token=80000,
              top_p=0.9)


if __name__ == '__main__':
    """
    test 1 qa
    """
    """
    test 2 qg
    """
    # query = '云天明送给程心的行星叫什么？'
    # print(es_search('云天明送给程心的行星叫什么？', app.config['ES_INDEX_DOC_ThreeBody']))
    # cmd_line_interactive()

    app.run('0.0.0.0', 5061, debug=True)
