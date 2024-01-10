import os
import streamlit as st
from google.cloud import storage
import openai
from google.cloud.sql.connector import Connector
import sqlalchemy

import os, tempfile
import streamlit as st, pinecone
from langchain.llms.openai import OpenAI
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
#from langchain.document_loaders import TextDocument

from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


class Document:
    def __init__(self, page_content):
            self.page_content = page_content
            self.metadata = {}

# Set the environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/app/servicekey.json'

# Set the project ID and bucket name
project_id = 'virtual-sylph-384316'
bucket_name = 'damg7245-assignment-7007'

# Initialize the Google Cloud Storage client
storage_client = storage.Client(project=project_id)
bucket = storage_client.bucket(bucket_name)

# Get list of files from GCS
files = [blob.name for blob in bucket.list_blobs()]

# Extract company names and years from files
companies = []
company_years = {}
for file in files:
    company, year = file.split(':')
    companies.append(company)
    if company not in company_years:
        company_years[company] = []
    company_years[company].append(year)

# Remove duplicates
companies = list(set(companies))

#page title - earnings statement
st.markdown("<h1 style='text-align: center; color: white;'>Earnings Statement</h1>", unsafe_allow_html=True)

# User selection
st.markdown("<h2 style='font-weight: bold;'>Select a company</h2>", unsafe_allow_html=True)
company = st.selectbox("Company", companies)

st.markdown("<h2 style='font-weight: bold;'>Select a year</h2>", unsafe_allow_html=True)
year = st.selectbox("Year", company_years[company])

# Get data from GCS
file = f"{company}:{year}"
blob = bucket.blob(file)
data = blob.download_as_text()

# Truncate data to a maximum of 2048 tokens
max_tokens = 2048
data_tokens = data.split()
if len(data_tokens) > max_tokens:
    data = ' '.join(data_tokens[:max_tokens])

# Streamlit app
st.subheader('Summarize the Transcript with LangChain & Pinecone')

# Get OpenAI API key, Pinecone API key, environment and index, and the source document input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", type="password")
    pinecone_api_key = st.text_input("Pinecone API key", type="password")
    pinecone_env = st.text_input("Pinecone environment")
    pinecone_index = st.text_input("Pinecone index name")

    if pinecone_api_key and pinecone_env and pinecone_index:
            # Initialize Pinecone connection
            pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

            try:
                # Create Pinecone index
                if pinecone_index not in pinecone.list_indexes():
                    # Create Pinecone index
                    st.write("Creating Pinecone index. This may take a few minutes...")
                    pinecone.create_index(
                        name=pinecone_index,
                        dimension=1536,
                        metric='cosine'
                    )
                    st.write("Pinecone index created.")
            except Exception as e:
                st.write(f"Error while creating Pinecone index: {e}")


if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key or not pinecone_api_key or not pinecone_env or not pinecone_index:
        st.warning(f"Please provide the missing fields.")
    else:

        # Create a Document object from the data string
        doc = Document(page_content=data)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectordb = Pinecone.from_documents([doc], embeddings, index_name=pinecone_index)
        # Initialize the OpenAI module and load the summarize chain
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        chain = load_summarize_chain(llm, chain_type="stuff")
        search = vectordb.similarity_search(" ")
        
        summary = chain.run(input_documents=search, question="Write a concise summary within 200 words.")
        st.success(summary)
            

# Initialize LangChain GQA
#Generative question and answering using langchain and pinecone in bold
st.markdown("<h2 style='font-weight: bold;'>Generative question and answering using LangChain and Pinecone</h2>", unsafe_allow_html=True) 
# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize Pinecone vector store
text_field = "text"
index = pinecone.Index(pinecone_index)
vectorstore = Pinecone(index, embeddings.embed_query, text_field)

llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo-16k', temperature=0.0)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Run a query
st.markdown("<h4 style='font-weight: bold;'>Ask a question about the transcript</h4>", unsafe_allow_html=True)
query = st.text_input("User query")
if query:
    result = qa.run(query)
    st.success(result)

#clear cache
if st.button("Query again"):
    st.cache_data.clear()