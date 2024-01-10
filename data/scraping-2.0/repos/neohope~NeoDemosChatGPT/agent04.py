#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
import json
import re
from langchain.agents import initialize_agent
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.agents import tool
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import CSVLoader

'''
以客户的场景，演示agent的用法
通过FAISS和VectorDBQA增强Tool的功能
强化订单查询功能，通过json串进行匹配
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

@tool("Search Order")
def search_order(input:str)->str:
    # 注意这里的promot，如果没有告知AI订单不存在时如何处理，则容易得到非预期答案
    """一个帮助用户查询最新订单状态的工具，并且能处理以下情况： 
    1. 在用户没有输入订单号的时候，会询问用户订单号 
    2. 在用户输入的订单号查询不到的时候，会让用户二次确认订单号是否正确"""

    pattern = r"\d+[A-Z]+" 
    match = re.search(pattern, input)

    order_number = input
    if match: 
        order_number = match.group(0) 
    else: 
        return "请问您的订单号是多少？" 
    
    if order_number == ORDER_1: 
        return json.dumps(ORDER_1_DETAIL) 
    elif order_number == ORDER_2: 
        return json.dumps(ORDER_2_DETAIL) 
    else: 
        return f"对不起，根据{input}没有找到您的订单"

@tool("Recommend Product")
def recommend_product(input: str) -> str:
    """"useful for when you need to search and recommend products and recommend it to the user"""
    return product_chain.run(input)

@tool("FAQ")
def faq(intput: str) -> str:
    """"useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return faq_chain.run(intput)

def create_orders():
    ORDER_1 = "20230101ABC"
    ORDER_2 = "20230101EFG"

    ORDER_1_DETAIL = {
        "order_number": ORDER_1,
        "status": "已发货",
        "shipping_date" : "2023-01-03",
        "estimated_delivered_date": "2023-01-05",
    } 

    ORDER_2_DETAIL = {
        "order_number": ORDER_2,
        "status": "未发货",
        "shipping_date" : None,
        "estimated_delivered_date": None,
    }

    return ORDER_1,ORDER_2,ORDER_1_DETAIL,ORDER_2_DETAIL

def create_faq_chain(llm):
    loader = TextLoader('./data/ecommerce_faq.txt')
    documents = loader.load()
    text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(texts, embeddings)

    faq_chain = VectorDBQA.from_chain_type(llm=llm, vectorstore=docsearch, verbose=True)
    return faq_chain

def create_product_chain(llm):
    product_loader = CSVLoader('./data/ecommerce_products.csv')
    product_documents = product_loader.load()
    product_text_splitter = CharacterTextSplitter(chunk_size=1024, separator="\n")
    product_texts = product_text_splitter.split_documents(product_documents)

    product_search = FAISS.from_documents(product_texts, OpenAIEmbeddings())
    product_chain = VectorDBQA.from_chain_type(llm=llm, vectorstore=product_search, verbose=True)
    return product_chain

def create_agent(llm):
    tools = [search_order,recommend_product, faq]
    agent = initialize_agent(tools, llm=llm, agent="zero-shot-react-description", verbose=True)
    return agent


if __name__ == '__main__':
    get_api_key()
    llm = OpenAI(temperature=0)
    ORDER_1,ORDER_2,ORDER_1_DETAIL,ORDER_2_DETAIL= create_orders()
    faq_chain = create_faq_chain(llm)
    product_chain = create_product_chain(llm)
    agent = create_agent(llm)

    question = "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？"
    answer = agent.run(question)
    print(answer)
