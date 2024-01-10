# https://github.com/nicknochnack/LangchainDocuments
# pip install langchain pypdf PyCryptodome chromadb

# Import os to set API key
import os
# Bring in streamlit for UI/app interface
import streamlit as st
import pathlib

# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = 'yourapikeyhere'

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)

# Initialize the parameters
agent_executor = None
store = None

def loadPDF(f):
    # Create and load PDF Loader
    loader = PyPDFLoader(f)
    # Split pages from pdf 
    pages = loader.load_and_split()
    # Load documents into vector database aka ChromaDB
    store = Chroma.from_documents(pages, collection_name='annualreport')

    # Create vectorstore info object - metadata repo?
    vectorstore_info = VectorStoreInfo(
        name="annual_report",
        description="a banking annual report as a pdf",
        vectorstore=store
    )
    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )
    return agent_executor, store

st.title('🦜🔗 ChatGPT PDF File Q&A App')

# Create a text input box for the OpenAI API key
st.header('1. Input your OpenAI API here')
apikey = st.text_input('Input your OpenAI API here')
st.write("Get your OpenAI API key here [link](https://platform.openai.com/account/api-keys)")
# If the user hits enter
if apikey:
    os.environ['OPENAI_API_KEY'] = apikey
    #st.write(os.environ['OPENAI_API_KEY'] )
    
st.header('2. Choose a PDF file')
uploaded_file = st.file_uploader("Choose a file","PDF")
if uploaded_file is not None:
    #data = uploaded_file.getvalue()     #.decode('utf-8')
    #parent_path = pathlib.Path(__file__).absolute().resolve() 
    #st.write(parent_path)
    #parent_path = pathlib.Path(__file__).parent.parent.resolve() 
    parent_path = pathlib.Path(__file__).parent.resolve() 
    save_path = os.path.join(parent_path, "docs")
    complete_name = os.path.join(save_path, uploaded_file.name)
    #complete_name = os.path.join(parent_path, uploaded_file.name)
    st.write(complete_name)
    #agent_executor, store =loadPDF(uploaded_file.name)
    agent_executor, store =loadPDF(complete_name)

#uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
#for uploaded_file in uploaded_files:
#    bytes_data = uploaded_file.read()
#    st.write("filename:", uploaded_file.name)
#    st.write(bytes_data)

# Create a text input box for the user
st.header('3. Input your prompt here')
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    if (agent_executor != None) and (store != None) and (os.environ['OPENAI_API_KEY'] != 'yourapikeyhere'):
        # Then pass the prompt to the LLM
        response = agent_executor.run(prompt)
        # ...and write it out to the screen
        st.write(response)

        # With a streamlit expander  
        with st.expander('Document Similarity Search'):
            # Find the relevant pages
            search = store.similarity_search_with_score(prompt) 
            # Write out the first 
            st.write(search[0][0].page_content) 
    else:
       st.write("Please select PDF file first")
