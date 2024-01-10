from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.milvus import Milvus

from lib.prompter import enriched_prompt

embeddings = OpenAIEmbeddings(disallowed_special=())
db = Milvus(embedding_function=embeddings)
docs = db.similarity_search("how do I load a file?")
for doc in docs:
    print(doc)
rsp = enriched_prompt("how do I load a file?")
print(rsp["answer"])
print(rsp)
