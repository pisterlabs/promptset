from langchain.document_loaders import UnstructuredPDFLoader,OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma,Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv('.env')

# 读取环境变量
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 用于加载pdf文件的方法
loader = UnstructuredPDFLoader('../book.pdf')
# 加载数据, 这里的数据就是普遍的文本数据, 其中pdf内容位于data[0].page_content中
data = loader.load()
content = data[0].page_content
print(f'本书中共有{len(content)}字')
"""
这行代码的含义是定义了一个 RecursiveCharacterTextSplitter 文本分割器，
用于将文本拆分为固定长度的块，每个块的长度为 1000 个字符，
分块时相邻两个块之间没有重叠部分（即 chunk_overlap=0）。
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
# 将文本拆分为块，注意这里只能对原始data进行拆分，不能对content进行拆分
texts = text_splitter.split_documents(data)
print(f'一共被拆分为{len(texts)}块')
docsearch = Chroma.from_documents(texts, embeddings)
# query = "这本书的内容是啥？"
# docs = docsearch.similar_documents(query, include_metadata=True)
# llm = OpenAI(temperature=0,openai_api_key="apChPf5QkF7DFMOmndW8T3BlbkFJ8w7c1Q7d7rmt379Pz230")
# chain = load_qa_chain(llm=llm,chain_type="stuff")
# chain.run(input_document=docs, question=query)
