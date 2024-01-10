import streamlit as st
import datetime
import time
from langchain.llms.ollama import Ollama
from langchain.chat_models import ChatOllama
import langchain.document_loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.gpt4all import GPT4AllEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.agents.initialize import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import Tool
from langchain.tools import BaseTool
from langchain.tools.ddg_search import DuckDuckGoSearchRun
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain.utilities.wikipedia import WikipediaAPIWrapper
import langchain


ollama = ChatOllama(base_url='http://192.168.1.81:11434', model='dolphin2.2-mistral', temperature=0.1, streaming=True)

set_llm_cache(InMemoryCache())


### CREATE VECTORSTORE FUNCTION

def db_lookup():
    try:
        if url is not None:
            loader = langchain.document_loaders.WebBaseLoader(url)

            documents = loader.load()

            len(documents)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

            texts = text_splitter.split_documents(documents)

            len(texts)

            persist_directory = "./vectorstores/db/"

            embeddings = GPT4AllEmbeddings()

        
            vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)

            vectordb.persist()
            vectordb = None

    except:
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
    #        st.write(pdf_reader)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
    #        st.write(text)

    #        len(pdf_documents)
        
            pdf_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    #        st.write(pdf_text_splitter)

            pdf_texts = pdf_text_splitter.split_text(text=text)

            len(pdf_texts)

    #        st.write(pdf_splits)

            persist_directory = "./vectorstores/db/"

            pdf_embeddings = GPT4AllEmbeddings()
        
            pdf_vectordb = Chroma.from_texts(pdf_texts, embedding=pdf_embeddings, persist_directory=persist_directory)

            pdf_vectordb.persist()
            pdf_vectordb = None


    
# Sidebar Contents

with st.sidebar:
    st.sidebar.title('ANTLATT.com')
    st.sidebar.header('Add More Data to the Database')

##SIDEBAR PDF INPUT    
    pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf", disabled=False)

###SIDEBAR URL INPUT                
    url = st.sidebar.text_input('Enter a URL', placeholder="enter url here", disabled=False)
    with st.form('myform2', clear_on_submit=True):
        
        persist_directory = "/vectorstores/db"

        submitted = st.form_submit_button('Submit', disabled=not(url or pdf))
        if submitted:
            with st.spinner('Creating VectorStore, Saving to Disk...'):
                db_lookup()
                with st.success('Done!'):
                    st.write('VectorStore Created and Saved to Disk')

    
    st.markdown('''
    ## About
    This is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io)
    - [Langchain](https://python.langchain.com)
    - [Ollama](https://ollama.com)
    - [Mistral-7b](https://huggingface.co/illuin/mistral-7b)

    ''')
    add_vertical_space(5)
    st.write('Made by [antlatt](https://www.antlatt.com)')


### MAIN PAGE CONTENTS

st.title('ANTLATT.com')
st.header('Chat with Your Documents')


### Chat App
user_prompt = st.chat_input('Enter your message:', key="user_prompt")
if user_prompt:
    st.write(f'You: {user_prompt}'
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Send a Chat Message to the AI Assistant"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        persist_directory = "./vectorstores/db/"
        embeddings = GPT4AllEmbeddings()
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        retriever = vectordb.as_retriever()
        docs = retriever.get_relevant_documents(prompt)
        len(docs)
        retriever = vectordb.as_retriever(search_kwags={"k": 3})
        retriever.search_type = "similarity"
        retriever.search_kwargs = {"k": 3}
        
        qachain = RetrievalQA.from_chain_type(ollama, chain_type="stuff", retriever=retriever, return_source_documents=False)



### ChatOllama_Agent

        llm = ChatOllama(base_url='http://192.168.1.81:11434', model='dolphin2.2-mistral', temperature=0.1, streaming=True)
        tools = [
                        Tool(
                            name="chat",
                            func=ChatOllama,
                            description="Useful for chatting with the AI in general conversation."
                            
                        ),
                        Tool(
                            name="ddg_search",
                            func=DuckDuckGoSearchRun,
                            description="A search engine. Useful for when you need to answer questions about current events. Input should be a search query."
                        ),
                        Tool(
                            name="wikipedia_search",
                            func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
                            description="An online encyclopedia. Search Wikipedia for a query"
                        ),
                        Tool(
                            name="vectorstore_search",
                            func=qachain,
                            description="Search the local Database for a query"
                        )
                ]
        ollama_agent = initialize_agent(tools,
                                llm,
                                agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                verbose=True,
                                handle_parsing_errors=True)
        


        message_placeholder = st.empty()
        full_response = ollama_agent.run(prompt)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})



### Chat App End

#if __name__ == "__main__":
#    main()