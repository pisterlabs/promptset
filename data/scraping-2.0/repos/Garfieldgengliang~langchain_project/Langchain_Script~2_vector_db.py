# Data loader
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("ftMLDE-2021.pdf")
pages = loader.load_and_split()

print(pages[1].page_content)

# text spliter
import re, wordninja
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

openai.api_base = "https://api.fe8.cn/v1"
openai.api_key = os.getenv('OPENAI_API_KEY')


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


from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,  # 思考：为什么要做overlap
    length_function=len,
    add_start_index=True,
)

paragraphs = text_splitter.create_documents([preprocess(pages[3].page_content)])
for para in paragraphs:
    print(para.page_content)
    print('-------')


# doctran
from langchain.document_transformers import DoctranTextTranslator

translator = DoctranTextTranslator(
    openai_api_model="gpt-3.5-turbo", language="Chinese"
)

translated_document =   await translator.atransform_documents([pages[3]])

print(translated_document[0].page_content)

# retrieve and question answer 用传统的关键字检索
from langchain.retrievers import TFIDFRetriever  # 最传统的关键字加权检索
from langchain.text_splitter import RecursiveCharacterTextSplitter
import wordninja, re

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=400,
    length_function=len,
    add_start_index=True,
)

# 取一个有信息量的章节（Introduction: 第2-3页）
paragraphs = text_splitter.create_documents(
    [preprocess(d.page_content) for d in pages[2:20]]
)

user_query = "What's the process of MLDE?"

retriever = TFIDFRetriever.from_documents(paragraphs)
docs = retriever.get_relevant_documents(user_query)

print(docs[2].page_content)
len(docs)


# hardcode query from doc，这时我们加上chatGPT的能力，会发现回答变得更流畅了
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

# 换一个问题
user_query = "What are the evaluation metric for MLDE process?"

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

# 向量检索,
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(paragraphs, embeddings) #Facebook的开源向量检索引擎

user_query = "What are the evaluation metric for MLDE process?"

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

# 向量数据库中文输入
user_query = "MLDE方法的评测标准是什么"

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

# text embedding
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings() # 默认是text-embedding-ada-002
text = "这是一个测试"
document = "测试文档"
query_vec = embeddings.embed_query(text)
doc_vec = embeddings.embed_documents([document])

print(len(query_vec))
print(query_vec[:10])  # 为了展示方便，只打印前10维
print(len(doc_vec[0]))
print(doc_vec[0][:10])  # 为了展示方便，只打印前10维


# vector stores parent document retriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
import faiss

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=60,
    length_function=len,
    add_start_index=True,
)

embedding_size = 1536 # OpenAIEmbeddings的维度
index = faiss.IndexFlatL2(embedding_size) # 精准检索
embedding_fn = OpenAIEmbeddings().embed_query
# 构造向量数据库
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

# 文档存储
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=text_splitter,
)

retriever.add_documents(pages[:30], ids=None)

user_query = "What are the evaluation metric for MLDE process?"
sub_docs = vectorstore.similarity_search(user_query)
print("===段落===")
print(sub_docs[0].page_content)

retrieved_docs = retriever.get_relevant_documents(user_query)
print("===文档===")
print(retrieved_docs[0].page_content)



