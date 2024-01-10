import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from config.setting import Setting
from config.pinecone_setting import index_name

Setting()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '../../data/jianzhudiyike.pdf')
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_documents(pages, embeddings, index_name=index_name)

query = "建筑是什么"
docs = docsearch.similarity_search(query)
print(docs)
