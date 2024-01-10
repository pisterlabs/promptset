# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader, PythonLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

import argparse
import os
import sys
import time

import dotenv
 
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage


print_debug = False

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = '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'

def load_documents(repo_path, type):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob=f"**/**/*",
        suffixes=[type],
        parser=LanguageParser()
    )

    new_documents = loader.load()

    if not print_debug:
        return new_documents

    print(f"Found {len(new_documents)} {type} documents.")
    for new_document in new_documents:
        print(f"Document: {new_document.metadata}")
    
    return new_documents


def print_texts(texts):
    if not print_debug:
        return
    print(f"Found {len(texts)} texts.")
    i=0
    for text in texts:
        i = i+1
        print(f"{i}:{text.metadata}")


# Create instance of OpenAI LLM
model = "gpt-4"
temperature = 0.9
llm = ChatOpenAI(temperature=temperature, model=model, verbose=True, streaming=True)
embeddings = OpenAIEmbeddings()

# Create and load PDF Loader
#loader = PyPDFLoader('llm_context/annualreport.pdf')

working_dir = os.getcwd()

repo_path = working_dir

print(f"repo_path: {repo_path}")

documents = []

python_documents = load_documents(repo_path, ".py")

markdown_documents = load_documents(repo_path, ".md")

log_documents = load_documents(repo_path, ".log")

print(f"Found {len(python_documents)} python documents.")
print(f"Found {len(markdown_documents)} markdown documents.")
print(f"Found {len(log_documents)} log documents.")

chunk_size=1000
chunk_overlap=50
python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                            chunk_size=chunk_size, 
                                                            chunk_overlap=chunk_overlap)
markdown_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN,
                                                                    chunk_size=chunk_size,
                                                                    chunk_overlap=chunk_overlap)

log_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN,
                                                                    chunk_size=chunk_size,
                                                                    chunk_overlap=chunk_overlap)
                                                            

texts = python_splitter.split_documents(python_documents)


texts += markdown_splitter.split_documents(markdown_documents)


texts += log_splitter.split_documents(log_documents)
print_texts(texts)



# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(texts, embeddings, collection_name='RIHEVNAA')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="RIHEVNAA_GitHub",
    description="Github repo for RI-HEVNAA",
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
st.title('RIHEVNAA Dev Chatbot')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

k = 10

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)



    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt, k=k) 
        # Write out the first
        #st.write(search[0][0].page_content) 

        # Print out the first k
        for i in range(k):
            st.write(search[i][1])
            st.write(search[i][0].metadata)
            st.write(search[i][0].page_content)
        
        #agent_executor.
        #write out all of them
        #st.write(search)
