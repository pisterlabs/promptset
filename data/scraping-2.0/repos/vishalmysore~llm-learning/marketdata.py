from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pandas as pd
import os
loader = CSVLoader(file_path='news/stocks.csv')  
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])
chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
query = "how many ticker symbol are there?"
response = chain({"question": query})
print(response['result'])
query = "provide all unique ticker?"
response = chain({"question": query})
print(response['result'])
query = "provide name of company for each symbol"
response = chain({"question": query})
print(response['result'])
query = "what was the highest stock price as per sheet?"
response = chain({"question": query})
print(response['result'])
query = "do you see any voltality in any of the symbol?"
response = chain({"question": query})
print(response['result'])
query = "can you create a test scenario based on cucumber?"
response = chain({"question": query})
print(response['result'])