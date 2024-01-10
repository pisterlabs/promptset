import os
import time

from flask import Flask
from flask import render_template
from flask import request
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate

)


app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = ""

# 加载 youtube 频道
loader = DirectoryLoader('D:/IdeaProjects/document.ai/code/langchain/data/', glob='**/*.txt')
# loader = TextLoader('D:/IdeaProjects/document.ai/code/langchain/data/徐易容.txt', 'utf-8')
# 将数据转成 document
documents = loader.load()

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


# 初始化 openai embeddings
embeddings = OpenAIEmbeddings()
# Store document embeddings temporarily in Chroma vector database for matching queries later
db = FAISS.from_documents(docs, embeddings)

# 通过向量存储初始化检索器
retriever = db.as_retriever()

system_template = """
Use the following context to answer the user's question.
If you don't know the answer, say you don't, don't try to make it up. And answer in Chinese.
-----------
{context}
-----------
{chat_history}
你是小着客服，小着是一个我们公司的名称，注意，回答问题的时候，尽量用中文回答问题。
"""

# 构建初始 messages 列表，这里可以理解为是 openai 传入的 messages 参数
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template('{question}')
]

# 初始化 prompt 对象
prompt = ChatPromptTemplate.from_messages(messages)


# 初始化问答链
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1, max_tokens=256), retriever)


@app.route('/')
def hello_world():
    return render_template('index1.html')

chat_history = []
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search = data['search']
    # 开始发送问题 chat_history 为必须参数,用于存储对话历史
    result = qa({'question': search, 'chat_history': chat_history})
    chat_history.append((search, result['answer']))
    print(result['answer'])
    return {
        "code": 200,
        "data": {
            "search": search,
            "answer": result['answer'],
            "tags": "tag",
        },
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3002)