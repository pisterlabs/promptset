import configparser

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base=openai_api_base)
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

# 打印向量数据库中的文档数量
print(vectordb._collection.count())

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, openai_api_base=openai_api_base)

#创建memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(),
    memory=memory,
)

question = "不苦主要做了什么"
result = qa_chain({"question": question})
print(result["answer"])

question = "那他是哪一年毕业的"
result = qa_chain({"question": question})
print(result["answer"])

