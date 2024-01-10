from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# 创建语言模型
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# # 创建文本加载器
loader = TextLoader('../20report.txt', encoding='utf8')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma, 
    embedding=embeddings,
    text_splitter=text_splitter
).from_loaders([loader])
query = "可以跟我说一说如何实现中华民族伟大复兴吗？"
result = index.query(query)
print(result)