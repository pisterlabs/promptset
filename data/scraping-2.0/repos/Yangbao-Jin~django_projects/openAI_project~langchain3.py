
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("llama2.pdf")
pages = loader.load_and_split()

print(pages[0].page_content)

import re, wordninja

#预处理字符全都连在一起的行
def preprocess(text):
    def split(line):
        tokens = re.findall(r'\w+|[.,!?;%$-+=@#*/]', line)
        return [
            ' '.join(wordninja.split(token)) if token.isalnum() else token
            for token in tokens
        ]

    lines = text.split('\n')
    for i,line in enumerate(lines):
        if len(max(line.split(' '), key = len)) >= 20: 
            lines[i] = ' '.join(split(line))
    return ' '.join(lines)

from langchain.retrievers import TFIDFRetriever  # 最传统的关键字加权检索
from langchain.text_splitter import RecursiveCharacterTextSplitter
import wordninja, re

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=60,  
    length_function=len,
    add_start_index=True,
)

# 取一个有信息量的章节（Introduction: 第2-3页）
paragraphs = text_splitter.create_documents(
    [preprocess(d.page_content) for d in pages[2:4]]
)

user_query = "Does llama 2 have a dialogue version?"

retriever = TFIDFRetriever.from_documents(paragraphs)
docs = retriever.get_relevant_documents(user_query)

print(docs[0].page_content)


from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "你是问答机器人，你根据以下信息回答用户问题。\n" +
            "已知信息：\n{information}\n\nBe brief, and do not make up information."),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

llm = ChatOpenAI(temperature=0)
response = llm(
            template.format_messages(
                information=docs[0].page_content,
                query=user_query
            )
        )
print(response.content)


#我们换个问法
user_query = "Does llama 2 have a conversational variant?"

retriever = TFIDFRetriever.from_documents(paragraphs)
docs = retriever.get_relevant_documents(user_query)

print("===检索结果===")
print(docs[0].page_content)

response = llm(
            template.format_messages(
                information=docs[0].page_content,
                query=user_query
            )
        )

print("===回答===")
print(response.content)


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings() 
db = FAISS.from_documents(paragraphs, embeddings) #Facebook的开源向量检索引擎

user_query = "Does llama 2 have a conversational variant?"

docs = db.similarity_search(user_query)
print("===检索结果===")
print(docs[0].page_content)

response = llm(
            template.format_messages(
                information=docs[0].page_content,
                query=user_query
            )
        )

print("===回答===")
print(response.content)


#你甚至可以跨语言检索
user_query = "llama 2有对话式的版本吗"

docs = db.similarity_search(user_query)
print("===检索结果===")
print(docs[0].page_content)

response = llm(
            template.format_messages(
                information=docs[0].page_content,
                query=user_query
            )
        )

print("===回答===")
print(response.content)

