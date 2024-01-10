import streamlit as st
import openai
import os
# azure.py
# Import necessary libraries
import streamlit as st
#import nltk
#nltk.download('punkt')
import tiktoken

# Replace these imports with your actual backend code
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Set up your OpenAI API key
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Initialize Streamlit
st.title("Boston City Code Chatbot")
st.subheader("Welcome to the Boston City Code Chatbot! Your one stop shop for all things Boston law.")

# Create a text input field for user queries
user_input = st.text_input("Please input your question below:")


if user_input:
    # Your existing backend code
    query = user_input  # Assuming the user input is the query

# Summarization Feature
st.subheader("Text Summarization")
user_input_text = st.text_area("Enter the text you want to summarize:")

def summarize_user_input(input_text):
    if len(input_text) > 1000:
        input_text = input_text[:1000] + '...'

    prompt = f"Please provide a brief and concise summary of the following text:\n\n'{input_text}'\n\nSummary:"
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.7,
            max_tokens=30
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error during API call: {e}")
        return None

if st.button("Summarize"):
    summarized_text = summarize_user_input(user_input_text)
    if summarized_text:
        st.write("Summarized Text:", summarized_text)
    else:
        st.write("No summary generated.")

    
    # Replace this block with your existing backend code
    dataset_corpus_path = "Short Boston Code.pdf"
    
    pdf_loader = PyPDFDirectoryLoader(dataset_corpus_path)
    documents = pdf_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100
        )
    
    chunks = pdf_loader.load_and_split(text_splitter)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, temperature = 0.3)
    db = FAISS.from_documents(chunks, embeddings)
    
    chain = load_qa_chain(OpenAI(openai_api_key=openai_api_key), chain_type="stuff")
    
    docs = db.similarity_search(query, k=2)
    
    result = chain.run(input_documents=docs, question=query)
    
    # Display the result
    st.write(result)  # Modify this to display the result as needed

