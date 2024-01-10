# -*- coding: utf-8 -*-
"""
https://python.langchain.com/docs/modules/data_connection/vectorstores/
https://www.langchain.com.cn/modules/indexes/vectorstores

One of the most common ways to store and search over unstructured data is to embed it and 
store the resulting embedding vectors, and then at query time to embed the unstructured query and 
retrieve the embedding vectors that are 'most similar' to the embedded query. 
A vector store takes care of storing embedded data and performing vector search for you.
"""
import os
import torch as th
from langchain.embeddings import (OpenAIEmbeddings, HuggingFaceEmbeddings)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import (Chroma, FAISS)


print(th.cuda.get_device_name())  # NVIDIA GeForce GTX 1080 Ti
device = th.device("cuda" if th.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/LangChain"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 入门指南
with open('../../state_of_the_union.txt') as f:
    state_of_the_union = f.read()
    
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)
 
# embeddings_model = OpenAIEmbeddings()
checkpoint = "all-mpnet-base-v2"
embeddings_model = HuggingFaceEmbeddings(
    model_name=os.path.join(path_model, checkpoint),
    cache_folder=os.path.join(path_model, checkpoint),
    # model_kwargs={"device": "gpu"},
    # encode_kwargs={"normalize_embeddings": False}
    )
db = Chroma.from_texts(texts=texts, embedding=embeddings_model)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

print(docs[0].page_content)

# ----------------------------------------------------------------------------------------------------------------
# 添加文本
db.add_texts(["Ankush went to Princeton"])
query = "Where did Ankush go to college?"
docs = db.similarity_search(query)
 
# ----------------------------------------------------------------------------------------------------------------
# 来自文档
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.create_documents(texts=[state_of_the_union], 
                                           metadatas=[{"source": "State of the Union"}])
db = Chroma.from_documents(documents=documents, embedding=embeddings_model)
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

# ----------------------------------------------------------------------------------------------------------------
# FAISS
loader = TextLoader('../../../state_of_the_union.txt')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings_model = OpenAIEmbeddings()
db = FAISS.from_documents(documents=documents, embedding=embeddings_model)
 
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
docs[0].page_content

docs_and_scores = db.similarity_search_with_score(query)
docs_and_scores[0]

# 使用 similarity_search_by_vector 可以搜索与给定嵌入向量类似的文档，该函数接受嵌入向量作为参数而不是字符串
embedding_vector = embeddings_model.embed_query(query)
docs_and_scores = db.similarity_search_by_vector(embedding_vector)

db.save_local("faiss_index")
new_db = FAISS.load_local(folder_path="faiss_index", embeddings=embeddings_model)
 
db1 = FAISS.from_texts(["foo"], embeddings_model)
db2 = FAISS.from_texts(["bar"], embeddings_model)
 
db1.docstore._dict
db2.docstore._dict
db1.merge_from(db2)




