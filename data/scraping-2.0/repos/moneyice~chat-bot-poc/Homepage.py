import streamlit as st
import tcvectordb
from tcvectordb.model.collection import Embedding, UpdateQuery
from tcvectordb.model.document import Document, Filter, SearchParams
from tcvectordb.model.enum import FieldType, IndexType, MetricType, EmbeddingModel
from tcvectordb.model.index import Index, VectorIndex, FilterIndex, HNSWParams, IVFFLATParams
from tcvectordb.model.enum import FieldType, IndexType, MetricType, ReadConsistency
import pandas as pd
from langchain.chat_models import ErnieBotChat
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import re
import json


# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn



def askButtonQuestion(question):
    answerThisQuestionFromButton(question)
    # with st.chat_message("user"):
    #     st.markdown(question+"=====================")

def initVectorDB():
    global coll
    client = tcvectordb.VectorDBClient(
        url='http://lb-jiy3uhlz-ykqd35onj0vwmflt.clb.ap-shanghai.tencentclb.com:50000', username='root',
        key='bqEC5oNObIWWGE26OYTnsB3XVsfPNwhtoob9pPLt', read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
        timeout=300)
    # 指定写入原始文本的数据库与集合
    db = client.database('xiaoe')
    coll = db.collection('questions2')

def searchVectorDBForNearestQuestions(prompt):
    myQuestion = prompt
    doc_lists = coll.searchByText(
        embeddingItems=[myQuestion],
        # filter=Filter(Filter.In("bookName",["三国演义", "西游记"])),
        params=SearchParams(ef=200),
        limit=8,
        retrieve_vector=False,
        output_fields=[]
    )
    return doc_lists

def queryAnswerFromDB(response):
    match = re.search(r'\d{5}', response)
    if match:
        idStr = match.group()
        if idStr:
            print("答案对应的id是: " + idStr)
            doc_lists = coll.searchById(
                document_ids=[idStr],
                # filter=Filter(Filter.In("bookName",["三国演义", "西游记"])),
                params=SearchParams(ef=200),
                limit=1,
                retrieve_vector=False,
                output_fields=[]
            )
            doc = doc_lists[0][0]
            final_answer = doc["answer"]
            extend = doc["extend"]
            return doc

    return None






def askWenXin(chat,doc_lists):
    print("=====================================================")
    print("我的问题是: " + prompt)
    print("=====================================================")
    allQuestion = ""
    for i, docs in enumerate(doc_lists.get("documents")):
        for doc in docs:
            allQuestion = allQuestion + doc['text'] + "#" + doc['id'] + "\r\n"
    print("从向量数据库中找出8个最相近的问题: ")
    print(allQuestion)
    print("=====================================================")
    template = "针对问题 {input} , 在下列语句里面选择一个意思最接近的，找到后，只返回语句#号后面的数字，不要返回多余的话: \r\n" + allQuestion
    print("使用如下问题模板向文心一言提问: \r\n" + template)
    print("=====================================================")
    myQuestion = prompt
    prompt_template = PromptTemplate(input_variables=['input'], template=template)
    llmchain = LLMChain(llm=chat, prompt=prompt_template)
    response = llmchain.run(myQuestion)
    print("文心一言的回答是：")
    print(response)
    print("=====================================================")
    return response

def displayAnswerFromButton(doc):
    final_answer = doc["answer"]
    extend = doc["extend"]
    ai_message = AIMessage(content=final_answer)
    st.session_state["messages"].append(ai_message)
    if (extend != "[]"):
        parsed_data = json.loads(extend)
        for item in parsed_data:
            button = st.button(item['question'], on_click=askButtonQuestion,
                               args=[item['question']])
def displayAnswerFromInput(doc):
    final_answer = doc["answer"]
    extend = doc["extend"]
    ai_message = AIMessage(content=final_answer)
    st.session_state["messages"].append(ai_message)
    with st.chat_message("assistant"):
        st.markdown(ai_message.content, True)
        if (extend != "[]"):
            parsed_data = json.loads(extend)
            for item in parsed_data:
                button = st.button(item['question'], on_click=askButtonQuestion,
                                   args=[item['question']])

def answerThisQuestionFromButton(prompt):
    st.session_state["messages"].append(HumanMessage(content=prompt))
    doc_lists = searchVectorDBForNearestQuestions(prompt)
    response = askWenXin(chat, doc_lists)
    doc = queryAnswerFromDB(response)
    if doc:
        displayAnswerFromButton(doc)
    else:
        with st.chat_message("assistant"):
            st.markdown("没有找到对应的答案")


def answerThisQuestionFromInput(prompt):
    st.session_state["messages"].append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    doc_lists = searchVectorDBForNearestQuestions(prompt)
    response = askWenXin(chat, doc_lists)
    doc = queryAnswerFromDB(response)
    if doc:
        displayAnswerFromInput(doc)
    else:
        with st.chat_message("assistant"):
            st.markdown("没有找到对应的答案")



chat = None
global coll
model_name = 'ERNIE-Bot-4'
if "ERNIE_CLIENT_ID" in st.session_state and "ERNIE_CLIENT_SECRET" in st.session_state and st.session_state["ERNIE_CLIENT_ID"]!="" and st.session_state["ERNIE_CLIENT_SECRET"]!="":
    chat = ErnieBotChat(ernie_client_id=st.session_state["ERNIE_CLIENT_ID"],
                        ernie_client_secret=st.session_state["ERNIE_CLIENT_SECRET"], model_name=model_name,
                        temperature=0.01)

    st.set_page_config(page_title="Welcome to 问答助手", layout="wide")
    st.title("🤠 Welcome to 问答助手")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    initVectorDB()
    if chat:
        with st.container():
            st.header("开始提问")

            for message in st.session_state["messages"]:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.markdown(message.content,True)
            prompt = st.chat_input("Type something...")
            if prompt:
                answerThisQuestionFromInput(prompt)
else:
    with st.container():
        st.warning("请设置文心一言的密钥")
