from langchain.document_loaders import WebBaseLoader
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import os
from keys import OPENAI_API_KEY

# 本程序实现了RAG 和 memory，但是没有实现UI

# 设置密钥和代理，为了可以访问openAI 的API
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# 获取到数据
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
loader = UnstructuredWordDocumentLoader('my_data/18 tourist.docx')
data = loader.load()

# 对数据进行分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# 准备模型和memory
llm = ChatOpenAI()
# memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

retriever = vectorstore.as_retriever()
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

res = qa("广东有什么好玩的地方推荐吗？最好是清远的，推荐一个就够了")
print(res)
print(qa("能再详细的介绍一下这个地方吗？"))

# 这个问题感觉模型不能识别上下文
print(qa("请浏览我们之前的交流记录，告诉我我的上一个问题是什么"))

# 这两个问题说明能够识别上下文
print(qa("翻译为英文：我爱学习"))
print(qa("翻译为日文"))

