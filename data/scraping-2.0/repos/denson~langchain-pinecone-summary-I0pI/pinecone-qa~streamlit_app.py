import os, tempfile
import streamlit as st, pinecone
from langchain.llms.openai import OpenAI
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader



# Streamlit app
st.subheader('Generative Q&A with LangChain & Pinecone')
            
# Get OpenAI API key, Pinecone API key and environment, and source document input
with st.sidebar:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX")
    openai_api_key = st.text_input("OpenAI API key", type="password")
    pinecone_api_key = st.text_input("PINECONE_API_KEY", value="{}".format(PINECONE_API_KEY), type="password")        
    pinecone_env = st.text_input("PINECONE_ENVIRONMENT", value="{}".format(PINECONE_ENVIRONMENT))
    pinecone_index = st.text_input("PINECONE_INDEX", value="{}".format(PINECONE_INDEX))
source_doc = st.file_uploader("Upload source document", type="pdf", label_visibility="collapsed")
query = st.text_input("Enter your query")

if st.button("Submit"):
    # Validate inputs
    if not openai_api_key or not pinecone_api_key or not pinecone_env or not pinecone_index or not query:
        st.warning(f"Please upload the document and provide the missing fields.")

    elif source_doc:
        try:
            # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load_and_split()
            os.remove(tmp_file.name)
            
            # Generate embeddings for the pages, insert into Pinecone vector database, and expose the index in a retriever interface
            pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectordb = Pinecone.from_documents(pages, embeddings, index_name=pinecone_index)
            retriever = vectordb.as_retriever()

            # Initialize the OpenAI module, load and run the Retrieval Q&A chain
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
            response = qa.run(query)
            
            st.success(response)      
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        try:
            # initialize the vector data store
            vectordb = Pinecone.from_documents(pages, embeddings, index_name=pinecone_index)
            retriever = vectordb.as_retriever()

            # Initialize the OpenAI module, load and run the Retrieval Q&A chain
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
            response = qa.run(query)
            
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
