# Import libraries
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
import re
from openai import OpenAI
import time
import openai

def reformulate_query(original_query):
    try:
        response = openai.Completion.create(
            model="gpt-4-1106-preview",  # or the specific model you're using
            prompt="Reformulate this query into a more detailed and specific version: '" + original_query + "'",
            max_tokens=50  # Adjust the number of tokens as needed
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print("Error in generating response:", e)
        return original_query  # Fallback to original query in case of error


# Function to process the LLM response
def process_llm_response(llm_response):
    response = llm_response['result']
    sources = '\n\nSources:\n' + '\n'.join([source.metadata['source'] for source in llm_response["source_documents"]])
    return response + sources

# Import libraries and existing initialization code




# Backend logic encapsulated into a function
def get_backend_response(user_message):
    print("Initializing backend...")

    # Initialize the OpenAI client
    client = OpenAI()

    # Setup the directory for database
    persist_directory = 'db'
    embedding = OpenAIEmbeddings()

    # Specify the unique persist directory for the vectordb
    unique_persist_directory = '/Users/marian/Desktop/Hackatum/loganalysis/max/dora_the_logexplorer/db/9a81ee9dabdfbc3e9c94ad7cbaf6b79d'

    # Initialize the vector database
    vectordb = Chroma(persist_directory=unique_persist_directory, embedding_function=embedding)

    print("Vectordb initialized:", vectordb)

    # Create a retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    # Initialize the LLM
    llmm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llmm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True,
                                           verbose=True)

    conversation_history = ""

    # Append the detailed question to the conversation history
    conversation_history += "User (detailed): " + user_message + "\n"

    # Retrieve documents considering the conversation history and detailed question
    docs = retriever.get_relevant_documents(conversation_history)

    # Generate response considering the conversation history and detailed question
    llm_response = qa_chain(conversation_history)
    response = process_llm_response(llm_response)

    # Append the response to the conversation history
    conversation_history += "System: " + response + "\n"

    print("Response generated:", response)

    return response