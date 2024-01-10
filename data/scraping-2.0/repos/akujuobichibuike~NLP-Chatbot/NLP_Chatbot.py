import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from IPython.display import display
import ipywidgets as widgets

# Define a function to count tokens
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Function to split text into chunks
def split_text_into_chunks(text, max_chunk_length=1024):
    chunks = []
    for i in range(0, len(text), max_chunk_length):
        chunk = text[i:i + max_chunk_length]
        chunks.append(chunk)
    return chunks

with open(r"C:\Users\WCLENG-9\Desktop\ACCESS BANK PROJECT\Access Bank Textbook.txt", encoding='utf-8') as f:
    text = f.read()
    
# Initialize the GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Split the long text into chunks
text_chunks = split_text_into_chunks(text, max_chunk_length=1024)

# Initialize the OpenAIEmbeddings
api_key = "" # Replace with your own Openai API key
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,  # or your preferred chunk size
    chunk_overlap=24,
    length_function=count_tokens,
)

# Create a vector database (FAISS) from the chunks
chunks = text_splitter.create_documents(text_chunks)

# Load the QA chain for question-answering
chain = load_qa_chain(OpenAI(temperature=0, openai_api_key=api_key), chain_type="stuff")

# Initialize the OpenAI instance for conversational retrieval
openai_instance = OpenAI(openai_api_key=api_key, temperature=0.1)

# Create a vector database (FAISS) from the chunks
db = FAISS.from_documents(chunks, embeddings)

# Create a conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(openai_instance, db.as_retriever())

# Function to process user input and provide responses
def chat_with_bot():
    print("Welcome to NLP chatbot! Type 'exit' to stop.")
    while True:
        query = input("You: ")

        if query.lower() == 'exit':
            print("Thank you for using NLP chatbot!")
            break

        # Initialize an empty chat history for each question
        chat_history = []

        # Split the user's query into chunks if it's too long
        query_chunks = split_text_into_chunks(query, max_chunk_length=1024)

        for chunk in query_chunks:
            result = qa({"question": chunk, "chat_history": chat_history})
            print(f'Chatbot: {result["answer"]}')

# Call the chat_with_bot function to start the conversation
chat_with_bot()
