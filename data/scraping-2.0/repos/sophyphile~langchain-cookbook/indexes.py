# Indexes - Structuring documents so LLMs can work with them

# Document Loaders
# Easy ways to import data from other sources. Shared functionality with OpenAI Plugins, specifically retrieval oplugins
# Llama index...

# from langchain.document_loaders import HNLoader

# loader = HNLoader("https://news.ycombinator.com/item?id=34422627")
# data = loader.load()
# print(data)

# print (f"Found {len(data)} comments")
# print (f"Here's a sample:\n\n{''.join([x.page_content[:150] for x in data[:2]])}")

# Text Splitters
# Often times your document is too long (like a book) for your LLM. You need to split it up into chunks. Text splitters help with this.
# There are many ways one can split text into chunks; experiment with different ones to see which is best for you.

# from langchain.text_splitter import RecursiveCharacterTextSplitter # There are various types of text splitters.

# # This is a long document we can split up.
# with open('data/pgessay.txt') as f:
#     pg_work = f.read()

# print (f"You have {len([pg_work])} document")

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size for demonstration, normally he would do 1000.
#     chunk_size = 150,
#     chunk_overlap = 20
# )

# texts = text_splitter.create_documents([pg_work])
# print (f"You have {len(texts)} documents")

# print ("Preview:")
# # print (texts[0])
# print (texts[0].page_content, "\n")
# print (texts[1].page_content)

# Retrievers
# Easy ways to combine documents with language models
# There are many different types of retrievers, the most widely supported is the VectorStoreRetriever (because we are doing so much similarity search with embeddings)

# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings

# import os
# from dotenv import load_dotenv

# load_dotenv()

# openai_api_key = os.getenv("OPENAI_API_KEY")

# loader = TextLoader('data/pgessay.txt')
# documents = loader.load()

# # Get your splitter ready
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# # Split your docs into texts
# texts = text_splitter.split_documents(documents)

# # Get embedding engine ready
# embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# # Embed your texts
# db = FAISS.from_documents(texts, embeddings)

# # Init your retriever. Asking for just 1 document back.
# retriever = db.as_retriever()
# print(retriever)

# docs = retriever.get_relevant_documents("what types of things did the author want to build?")
# print("\n\n".join([x.page_content[:200] for x in docs[:2]]))

# Vector Stores
# Databases to store vectors. Most popular ones are Pinecone and Weaviate. More examples on OpenAIs retriever documentation. Chroma & FAISS are easy to work with locally.
# Conceptually, think of them as tables w/ a column for embeddings (vector) and a column for metadata. 
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings 

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

loader = TextLoader('data/pgessay.txt')
documents = loader.load()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# Get embedding engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

print (f"You have {len(texts)} documents")
embedding_list = embeddings.embed_documents([text.page_content for text in texts])

print (f"You have {len(embedding_list)} embeddings")
print (f"Here's a sample of one: {embedding_list[0][:3]}...")