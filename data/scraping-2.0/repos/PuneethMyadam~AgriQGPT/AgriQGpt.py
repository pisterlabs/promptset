import os
import sys

import streamlit as st

from apikey import apikey

import os
import pandas as pd
import matplotlib.pyplot as plt
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = apikey

# 1. vetorise the sales response csv data

loader = CSVLoader(file_path="C:/Users/Ydbrk/OneDrive/Documents/sample_data.csv", encoding='utf-8')
documents = loader.load()

# print(len(documents))

embeddings= OpenAIEmbeddings()
db= FAISS.from_documents(documents, embeddings)

#2.Funtion for similarity search

def retrive_info(query):
    similar_response= db.similarity_search(query, k=3)
    page_contents_array= [doc.page_content for doc in similar_response]
    #print(page_contents_array)
    return page_contents_array


#3. Setup a LLMChain & prompts

llm= ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo-16k-0613")
template= """

You are a Farming expert
{message}

"""

prompt= PromptTemplate(
    input_variables= ["message", "best_practices"],
    template= template
)
chain = load_qa_chain(OpenAI(temperature=0.4), chain_type="stuff")
chain1 = LLMChain(llm=llm, prompt= prompt)

# # 4. Retrival agumented generation
def generate_response(message):

# query = "I am a farmer and I do not have a proper fruit from my coconut plant what should I do?"
    docs = db.similarity_search(message)

    
    best_practice= retrive_info(message)
    response= chain.run(input_documents=docs, question=message, best_practice=best_practice)

    
    if response == "I don't know.":
        response1= chain1.run(message= message, best_practice=best_practice)
        return response1
    else:
        return response

# response= generate_response(message)
# print(response)

# Create QA chain to integrate similarity search with user queries (answer query from knowledge base)




# 5. App interface

def main():
    st.set_page_config(
        page_title= "AgriQGpt", page_icon="🌱"
    )

    st.header("AgriQGpt🌱")
    message= st.text_area("Customer Questions")

    if message:
        st.write("One Sec ......")

        result= generate_response(message)

        st.info(result)

if __name__== '__main__':
    main()




# import time

# def rate_limit_aware_request(question):
#     remaining_time = 20  # Initial wait time
#     while True:
#         try:
#             results = retrive_info(question)
#             return results
#         except RateLimitError:
#             print("Rate limit reached. Waiting for {} seconds...".format(remaining_time))
#             time.sleep(remaining_time)
#             remaining_time *= 2  # Increase wait time exponentially

# # Example usage
# question = "What is the time period to transplant paddy nursery?"
# results = rate_limit_aware_request(question)
# print(results)