#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
from langchain.agents import initialize_agent, Tool
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
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

def search_order(input: str) -> str:
    return "订单状态：已发货；发货日期：2023-01-01；预计送达时间：2023-01-10"

@tool("Recommend Product")
def recommend_product(input: str) -> str:
    """"useful for when you need to search and recommend products and recommend it to the user"""
    return product_chain.run(input)

@tool("FAQ")
def faq(intput: str) -> str:
    """"useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return faq_chain.run(intput)

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
    tools = [
    Tool(
        name = "Search Order",func=search_order, 
        description="useful for when you need to answer questions about customers orders"
    ),
    recommend_product, faq]

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    return agent


if __name__ == '__main__':
    get_api_key()
    llm = OpenAI(temperature=0)
    faq_chain = create_faq_chain(llm)
    product_chain = create_product_chain(llm)
    agent = create_agent(llm)

    question = "请问你们的货，能送到三亚吗？大概需要几天？"
    result = agent.run(question)
    print(result)

    question = "我想买一件衣服，想要在春天去公园穿，但是不知道哪个款式好看，你能帮我推荐一下吗？"
    answer = agent.run(question)
    print(answer)
