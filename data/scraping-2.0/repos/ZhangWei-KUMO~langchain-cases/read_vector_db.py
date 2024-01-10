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
# 从向量中获取数据
db = Chroma(embedding_function=embeddings,persist_directory='db')
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name='text-davinci-003',
                                            openai_api_key=OPENAI_API_KEY,
                                            max_tokens=1000,temperature=0), 
                                 chain_type="stuff", retriever=retriever)
query = "跟我介绍下这本书,写一篇1000字的概要"
result = qa.run(query)
print(result)