# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Bring in streamlit for UI/app interface
import streamlit as st

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

# Set API key for OpenAI Service
# Can substitute this out for other LLM providers
os.environ['OPENAI_API_KEY'] = 'yourapikeyhere'

# Create an instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)

# Create and load PDF loaders for each document
loader1 = PyPDFLoader('doc1.pdf')
pages1 = loader1.load_and_split()

loader2 = PyPDFLoader('doc2.pdf')
pages2 = loader2.load_and_split()

loader3 = PyPDFLoader('doc3.pdf')
pages3 = loader3.load_and_split()

# Combine the loaded pages into a single list
pages = pages1 + pages2 + pages3

# Create a vector store for each document and add it to the toolkit
store1 = Chroma.from_documents(pages1, collection_name='doc1')
store2 = Chroma.from_documents(pages2, collection_name='doc2')
store3 = Chroma.from_documents(pages3, collection_name='doc3')

toolkit = VectorStoreToolkit()
toolkit.add_vectorstore(store1)
toolkit.add_vectorstore(store2)
toolkit.add_vectorstore(store3)

# Create a vector store info object
vectorstore_info = VectorStoreInfo(
    name="Multiple Documents",
    description="A collection of multiple PDF documents",
    vectorstore=toolkit.get_vectorstore()
)

# Convert the vector store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

st.title('🦜🔗 GPT Investment Banker')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = toolkit.similarity_search_with_score(prompt)
        # Write out the first match
        if search:
            top_match = search[0][0]
            st.write(top_match.page_content)
        else:
            st.write("No matching documents found.")
