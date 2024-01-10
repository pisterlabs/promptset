# Question & Answering using Documents as Context
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Simple Q&A Example
# llm(your context + your question) = your answer

# from langchain.llms import OpenAI
# llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# context = """
# Rachel is 30 years old
# Bob is 45 years old
# Kevin is 65 years old
# """

# question = "Who is under 40 years old?"

# output = llm(context + question)

# # I strip the text to remove the leading and trailing whitespace
# print (output.strip())

# # This was pretty simple. The hard part comes when one needs to be selective about which data to put in the context. This field of study is called "document retrieval" and is tightly coupled with AI Memory.



# Using Embeddings
# Process: Split text, embed the chunks, put the embeddings in a DB, and then query them.
# The goal is to select relevant chunks of our long text, but which chunks to pull? The most popular method is to pull similar texts based off comparing vector embeddings.

from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# The vector-store we will use
from langchain.vectorstores import FAISS

# The LangChain component we'll use to get the documents ie vectors
from langchain.chains import RetrievalQA

# The easy document loader for text - load up text into documents
from langchain.document_loaders import TextLoader

# The embedding engine that will convert our text to vectors
from langchain.embeddings.openai import OpenAIEmbeddings


llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

loader = TextLoader('data/pgessay.txt')
doc = loader.load()
print (f"You have {len(doc)} document(s)")
print (f"You have {len(doc[0].page_content)} characters in that document")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)

# Get the total number of characters so we can see the average later
num_total_characters = sum([len(x.page_content) for x in docs])

print (f"Now you have {len(docs)} documents that have an average of {num_total_characters / len(docs):,.0f} characters (smaller pieces)")

# Get your embeddings engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Embed your documents and combine with the raw text in a pseudo DB. Note: This will make an API call to OpenAI.
docsearch = FAISS.from_documents(docs, embeddings)

# Create the retrieval engine
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

query = "What does the author describe as good work?"
print(qa.run(query))
# As a next step, one could hook this up to a cloud vector database, use a tool like metal and start managing documents with external data sources.



