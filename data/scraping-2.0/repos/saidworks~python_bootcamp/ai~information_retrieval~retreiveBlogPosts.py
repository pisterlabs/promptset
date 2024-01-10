from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

vector_store = Chroma.from_documents(
    documents=all_splits, embedding=GPT4AllEmbeddings())

question = input("Enter your question: ")
print("Searching for similar documents for question: ", question)
docs = vector_store.similarity_search(question)

print(len(docs))

print(type(docs[0]))
