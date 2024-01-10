from os import path
from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings

doc_directory = path.join(path.dirname(__file__), "vectordb/docs_500_100")
embeddings = HuggingFaceEmbeddings()
vectordb = Chroma(persist_directory=doc_directory,
                  embedding_function=embeddings)

docs_score = vectordb.similarity_search_with_score(
    "支持哪些格式？", include_metadata=True, k=3)

print(docs_score)

try:
    del vectordb
except Exception as e:
    print(e)
