'''
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2023 by Martain.AI, All Rights Reserved.
#
Description: # 
Author: # apollo2mars apollo2mars@gmail.com
################################################################################
'''

from data_connection import data_connection_load_web
from data_transformer import text_spliter
from data_vector_retireval import get_db
from constants import LangchainArgument

from langchain.llms import ChatGLM
from langchain.chains import RetrievalQA


data  = data_connection_load_web(LangchainArgument.web_baidu_http)
data= text_spliter(data)

for item in data:
    print(">>>")
    print(item)

db = get_db(data)


llm = ChatGLM(
    endpoint_url='http://127.0.0.1:8000',
    max_token=128,
    top_p=0.9
)
# 创建qa
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    verbose=True
)

# print(qa.)


response = qa.run({"prompt":'李扬说2024年是个什么年', "max_length":127})
print(type(response))
print(">>>" + response)