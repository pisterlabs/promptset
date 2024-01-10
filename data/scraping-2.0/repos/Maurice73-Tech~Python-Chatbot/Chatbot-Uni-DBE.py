#pip install -q langchain==0.0.150 pypdf pandas matplotlib tiktoken textract transformers openai faiss-cpu
#pip install langchain --upgrade

# Import libraries
import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import PyPDF2
import glob
from transformers import OpenAIGPTTokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Function to configure the environment
def configure():
    load_dotenv()

# Configure the environment
configure()

# Set up the OpenAI API key as an Environment Variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

directory = os.getcwd() + '/'

# Find all PDF files in the directory
pdf_files = glob.glob(directory + "*.pdf")

all_text = []

# Iterate through each PDF file
for pdf_file in pdf_files:
    # Open the PDF file in binary read mode
    with open(pdf_file, 'rb') as file:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(file)

        # Extract text from each page of the PDF
        for page in reader.pages:
            text = page.extract_text()
            if text:  # Check if the page contains text
                all_text.append(text)

# Combine all extracted text into one string
combined_text = '\n'.join(all_text)

# Write the extracted text to a file
with open('DBE_FactSheet_merged.txt', 'w', encoding='utf-8') as f:
    f.write(combined_text)

# Read the text from the file
with open('DBE_FactSheet_merged.txt', 'r') as f:
    text = f.read()

# Load the GPT-2 tokenizer
tokenizer = OpenAIGPTTokenizerFast.from_pretrained("gpt2")

# Function to count the number of tokens in a text using the GPT-2 tokenizer
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Split the text into chunks based on token count
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512, # Maximum number of tokens per chunk
    chunk_overlap  = 24, # Number of overlapping tokens between chunks
    length_function = count_tokens, # Function to count the number of tokens in a text
)

# Create the text chunks
chunks = text_splitter.create_documents([text])

# Generate the embeddings for the chunks
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

# Load the QA retrieval system with specific settings
chain = load_qa_chain(ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0), chain_type="stuff")
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.3), db.as_retriever())

# Initialize the chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] 

# Streamlit favicon
st.set_page_config(page_title="DBE Information Chatbot", page_icon=":robot_face:", layout="wide")

# Streamlit Interaction
st.image("DBE-Logo.png")

# Title
st.title('Digital Business Engineering - Information Chatbot')

# User input
query = st.chat_input("Please enter your question:")

# Function to display the conversation history
def display_conversation_history():
    for user_query, bot_response in st.session_state.chat_history:
        st.markdown(f"""
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                <span style="margin-left: 0px;"><strong>User:</strong> {user_query}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
            <div style="background-color: #ffffff; padding: 10px; border-radius: 5px;">
                <span style="margin-left: 0px;"><strong>ChatDBE_GPT:</strong> {bot_response}</span>
            </div>
            """, unsafe_allow_html=True)

# Handle the user's question
if query:
    # End the chatbot session if user types 'exit'
    if query.lower() == 'exit':
        st.write("Thank you for using the DBE Information Chatbot")
        st.session_state.chat_history.clear()  # Optionally clear the history upon exiting
    else:
        # Retrieve the answer for the user's question
        result = qa({"question": query, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((query, result['answer']))

        # Display the entire conversation history
        display_conversation_history()
