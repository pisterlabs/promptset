import json

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.vectorstores import Qdrant
from langchain_core.documents import Document
import keys
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def get_question(data_i):
    chat_template = ChatPromptTemplate.from_messages(
        [
            # SystemMessage(
            #     content=(
            #         "You receive the answer to a support request from a student. Please predict the most likely question that the student might have asked."
            #         "If possible, try to include the study program and the topic of the answer in your question."
            #         "Here is some potential relevant context: topic: {topic}, program: {program}"
            #     )
            # ),
            SystemMessagePromptTemplate.from_template(
                "You receive the answer to a support request from a student form the TUM School of Management."
                "They have Master and Bachelor study programs in Munich and Heilbronn."
                "Please predict the most likely question that the student might have asked."
                "If possible, try to include the study program and the topic of the answer in your question. 'General masters/bachelor' means that it applies to all master/bachelor programs which they offer."
                "Here is some potential relevant context: topic: {topic}, program: {program}"
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )

    llm = ChatOpenAI(model_name="gpt-4")
    # chat_template.format_messages(text=data_i['answer'], topic=data_i["topic"], program=data_i["program"])
    data_i["question"] = llm(chat_template.format_messages(text=data_i['answer'], topic=data_i["topic"], program=data_i["program"])).content
    print(".", end="", flush=True)
    
def add_questions(data):
    for data_i in data:
        if data_i["language"] == "EN" or data_i["language"] == "EN:":
            get_question(data_i)

            
file_names = ["standard_answers_master_munich_final.json", "standard_answers_master_heilbronn_final.json", "standard_answers_bachelor_munich_final.json", "standard_answers_bachelor_heilbronn_final.json"]

docs = []

def single_file(docs, file_name):
    with open(f"parsed_standard_reply_data/{file_name}", encoding="utf-8") as f:
        data = json.loads(f.read())
    add_questions(data)
    # this is a bug, should be question
    docs += [Document(page_content=data_i["question"], metadata={"source": file_name.split("/")[-1], **data_i}) for data_i in data if data_i["language"] == "EN" or data_i["language"] == "EN:"]

for i in file_names:
    single_file(docs, i)

    
# faq.json
with open('parsed_standard_reply_data/faq.json', encoding="utf-8") as f:
    data = json.loads(f.read())
# remove html tags
import html2text
h = html2text.HTML2Text()
h.ignore_links = False
data = [{ key: h.handle(value) for key, value in data_i.items()} for data_i in data]
docs += [Document(page_content=data_i["question"], metadata={"source": "faq.json", **data_i}) for data_i in data]

embeddings = OpenAIEmbeddings()
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=keys.url,
    prefer_grpc=True,
    api_key=keys.api_key,
    collection_name="tum4",
    force_recreate=True,
)