import os
import openai

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_transformers import (
    LongContextReorder, 
)
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from utils import getPages


# Get name of the PDF file
filename = input("Enter the path to your educational material: ")

# Get pages from the PDF
pages = getPages(filename)
print('Getting pages from PDF...')

# Set the type of embedding to use
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Make a list of the text from each page
texts = [page.page_content for page in pages]

# Create a retriever
retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 2}
)

# 
query = "How do I solve 1/x=2?"

# Get relevant documents ordered by relevance score
docs = retriever.get_relevant_documents(query)


print(len(docs))
print(f'The most similar document to the query "{query}" is "{docs[0].page_content}"')