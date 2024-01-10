from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

database = Chroma(
    persist_directory="./.data",
    embedding_function=embeddings
)

documents = database.similarity_search(
    "飛行車の最高速度は？"
)

print(f"ドキュメントの数: {len(documents)}")

for document in documents:
    print(f"ドキュメントの内容: {document.page_content}")