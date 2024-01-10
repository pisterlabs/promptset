import os, logging
# from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

logging.getLogger().setLevel(logging.INFO)

api_key = os.environ.get("OPENAI_API_KEY")

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.document_loaders import TextLoader

from langchain.agents import initialize_agent, tool, ConversationalChatAgent

llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)
loader = TextLoader('./data/ecommerce_faq.txt', encoding="utf8")
documents = loader.load()
text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)
doc_retriever = VectorStoreRetriever(vectorstore=docsearch)

## FAQ Chain
faq_chain = RetrievalQA.from_llm(llm=llm, retriever=doc_retriever, verbose=True)

# # test FAQ chain
# a1 = faq_chain.run("支持哪些支付方式")
# print(a1)

## Product Recommend Chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import CSVLoader

product_loader = CSVLoader('./data/ecommerce_products.csv', encoding="utf8")
product_documents = product_loader.load()
product_text_splitter = CharacterTextSplitter(chunk_size=1024, separator="\n")
product_texts = product_text_splitter.split_documents(product_documents)
product_search = FAISS.from_documents(product_texts, OpenAIEmbeddings())
product_retriever = VectorStoreRetriever(vectorstore=product_search)
product_chain = RetrievalQA.from_llm(llm=llm, retriever=product_retriever, verbose=True)

## test product chain
# a2 = product_chain.run("能推荐一件现在适合冬天穿的衣服给我么？")
# print(a2)

# define agent tools

# 1. search Order tool
import json

ORDER_1 = "20230101ABC"
ORDER_2 = "20230101EFG"

ORDER_1_DETAIL = {
    "order_number": ORDER_1,
    "status": "已发货",
    "shipping_date" : "2023-11-03",
    "estimated_delivered_date": "2023-11-05",
} 

ORDER_2_DETAIL = {
    "order_number": ORDER_2,
    "status": "未发货",
    "shipping_date" : None,
    "estimated_delivered_date": None,
}

import re

answer_order_info = PromptTemplate(
    template="请把下面的订单信息回复给用户： \n\n {order}?", input_variables=["order"]
)
answer_order_llm = LLMChain(llm=ChatOpenAI(temperature=0),  prompt=answer_order_info)


@tool("Search Order", return_direct=True)
def search_order(input:str)->str:
    """useful for when you need to answer questions about customers orders"""
    pattern = r"\d+[A-Z]+"
    match = re.search(pattern, input)

    order_number = input
    if match:
        order_number = match.group(0)
    else:
        return "请问您的订单号是多少？"
    if order_number == ORDER_1:        
        return answer_order_llm.run(json.dumps(ORDER_1_DETAIL))
    elif order_number == ORDER_2:
        return answer_order_llm.run(json.dumps(ORDER_2_DETAIL))
    else:
        return f"对不起，根据{input}没有找到您的订单"

@tool("FAQ")
def faq(intput: str) -> str:
    """"useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return faq_chain.run(intput)

@tool("Recommend Product")
def recommend_product(input: str) -> str:
    """"useful for when you need to search and recommend products and recommend it to the user"""
    return product_chain.run(input)

tools = [
    search_order,
    recommend_product,
    faq
]

def _handle_error(error) -> str:
    return str(error)[:50]

chatllm=ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_agent = initialize_agent(tools, chatllm, 
                                      agent="chat-conversational-react-description", 
                                      memory=memory, verbose=True,
                                      handle_parsing_errors=_handle_error)


if __name__ == '__main__':
    question1 = "我有一张订单，一直没有收到，能麻烦帮我查一下吗？"
    answer1 = conversation_agent.run(question1)
    print(answer1)

    question2 = "我的订单号是20230101ABC"
    answer2 = conversation_agent.run(question2)
    print(answer2)

    question3 = "你们的退货政策是怎么样的？"
    answer3 = conversation_agent.run(question3)
    print(answer3)

    question4 = "能推荐一件现在适合冬天穿的衣服给我么？用中文返回"
    answer4 = conversation_agent.run(question4)
    print(answer4)

    question5 = "请帮我简要介绍一下RAG的工作流程，用中文返回，尽量在400字以内"
    answer5 = conversation_agent.run(question5)
    print(answer5)