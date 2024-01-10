from langchain.document_loaders import JSONLoader
import json
from langchain.schema import Document

with open('./data/book_dump.json') as f:
    all_paragraphs = json.load(f)

docs = []
for p in all_paragraphs:
    d = Document(page_content=p["content"])
    docs.append(d)

from langchain.vectorstores import Chroma
# Using embedding models from OpenAI
# from langchain.embeddings import OpenAIEmbeddings
# vectorstore = Chroma.from_documents(documents=all_splits,embedding=OpenAIEmbeddings())
# Using local embedding models
from langchain.embeddings import HuggingFaceEmbeddings
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=HuggingFaceEmbeddings(),
    persist_directory="./chroma_store"
)

# single query
question = "It all starts with the universally applicable premise that people want to be understood and accepted. Listening is the cheapest, yet most effective concession we can make to get there. By listening intensely, a negotiator demonstrates empathy and shows a sincere desire to better understand what the other side is experiencing."
print(question)
docs = vectorstore.similarity_search(question)
len(docs)
# print(docs)
for doc in docs:
    print(doc)
    print("\n")
