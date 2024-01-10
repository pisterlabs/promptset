from os import path

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()  # model="text-embedding-ada-002"
doc_directory = path.join(path.dirname(__file__), "vectordb/docs_500_100")
vectordb = Chroma(persist_directory=doc_directory,
                  embedding_function=embeddings)

docs_score = vectordb.similarity_search_with_score(
    "支持什么格式？", include_metadata=True, k=3)
print(docs_score)
