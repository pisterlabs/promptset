from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""Helper functions"""

"""A function that can have an input passed to it, and it writes it to a file"""
def log_to_file(message):
    with open("log.txt", "a") as f:
        f.write(message + "\n")

"""Load in secrets from .env file, if cloning this repo then ensure you add your own .env file"""

load_dotenv()
api_key = os.getenv('API_KEY')

"""Basic setup for Langchain OpenAi API"""

llm = ChatOpenAI(openai_api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are writing a response that a 5 year old could understand."),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# response = chain.invoke({"input": "how can langsmith help with testing?"})

# print(response)

"""Using a Retriever and vector embeddings to pass in information from a document from the web"""

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key=api_key)

"""setting up the vector store"""
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = DocArrayInMemorySearch.from_documents(documents, embeddings)

"""setting up the chain so it can look up and pass along relevant information"""

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

"""invoke the chain"""
input = {"input": "What are the critical points mentioned about Langsmith?"}
response = retrieval_chain.invoke(input)
print(response["answer"])

log_to_file("Input: " + input["input"])
log_to_file("Response: " + response["answer"])
log_to_file("\n\n")



