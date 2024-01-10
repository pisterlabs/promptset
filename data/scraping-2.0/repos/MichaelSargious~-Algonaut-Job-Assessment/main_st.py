# Importing necessary libraries
import re
import os
import json
import urllib.request
import streamlit as st
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Load environment variables from a .env file for secure access to sensitive data like API keys
load_dotenv()

# Function to fetch and parse academic papers from the arXiv API
def fetch_papers():
    # URL for the arXiv API query, targeting papers with 'llama' in the title
    url = 'https://export.arxiv.org/api/query?search_query=ti:llama&start=0&max_results=70'

    # Making a request to the API and reading the response
    response = urllib.request.urlopen(url)
    data = response.read().decode('utf-8')

    # Parsing the XML response
    root = ET.fromstring(data)

    # Extracting paper information from the XML
    papers_list = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        paper_info = f"Title: {title}\nSummary: {summary}\n"
        papers_list.append(paper_info)

    return papers_list

# Function to save a list of strings to a JSON file
def save_list_to_json(list_of_strings, filename):
    with open(filename, 'w') as file:
        json.dump(list_of_strings, file)

# Function to load a list of strings from a JSON file
def load_list_from_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Function to initialize the QA chain using LangChain and FAISS
def initialize_qa_chain():
    # Getting the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Specifying the model for embeddings
    model = "text-embedding-ada-002"
    embeddings_model = OpenAIEmbeddings(model=model, api_key=openai_api_key)

    # Loading or creating a list of papers
    try:
        papers_list = load_list_from_json("papers_list.json")
    except FileNotFoundError:
        papers_list = fetch_papers()
        save_list_to_json(papers_list, "papers_list.json")

    # Loading or creating a FAISS index for efficient similarity search
    try:
        db = FAISS.load_local("faiss_index")
    except FileNotFoundError:
        db = FAISS.from_texts(papers_list, embeddings_model)
        db.save_local("faiss_index")

    # Configuring the retriever with a similarity score threshold
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.5})

    # Setting up the language model for question answering
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        api_key=openai_api_key
    )

    # Creating a prompt template for the QA system
    prompt_template = "{context}\n\nQuestion: {question}"
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["summaries", "question"],
            ),
        }
    )

    return qa

# Streamlit UI setup
st.title('Llama 2 QA System')

# Creating a text input field for user queries
query = st.text_input("Enter your query:")

# Defining action on button click
if st.button('Get Answer') and query:
    # Initialize QA chain
    qa = initialize_qa_chain()

    # Fetch response from the QA model
    response = qa(query)

    # Displaying the answer
    st.subheader("Answer:")
    st.write(response["result"])

    # Displaying the source papers
    st.subheader("Source Papers:")
    # Extracting titles from the source documents using regex
    titles = [re.search(r'Title: (.+?)\n', paper.page_content).group(1) for paper in response["source_documents"]]
    for e, title in enumerate(titles):
        st.write(f"{e + 1}- {title}")