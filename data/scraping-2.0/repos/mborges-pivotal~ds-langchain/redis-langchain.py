# Redis - Using redis-stack locally installed
#
# See https://python.langchain.com/docs/integrations/vectorstores/redis
#
#
# Issue - Got the error below using Redis cloud when storing the vector embeddings
# redis.exceptions.ResponseError: Vector index initial capacity 20000 exceeded server limit (511 with the given parameters)
import os
import getpass

os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

from langchain.vectorstores.redis import Redis

loader = TextLoader("state_of_the_union.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# PGVector needs the connection string to the database.
CONNECTION_STRING = "redis://localhost:6379"

db = Redis.from_documents(
    embedding=embeddings,
    documents=docs,
    redis_url=CONNECTION_STRING,
)

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
