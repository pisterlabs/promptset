import os
import re
import uuid
import openai
import promptlayer
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

MODEL = "gpt-3.5-turbo-16k"

# Remove HTML from sources
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_markdown(text):
    # Remove headers (e.g., # Header)
    text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
    # Remove bold/italic (e.g., **bold**, *italic*)
    text = re.sub(r'\*.*\*', '', text)
    # Remove links (e.g., [text](url))
    text = re.sub(r'\[.*\]\(.*\)', '', text)
    # Remove lists (e.g., - item)
    text = re.sub(r'- .*$', '', text, flags=re.MULTILINE)
    return text

def clean_text(text):
    text = remove_html_tags(text)
    text = remove_markdown(text)
    return text

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="Chat with Simon Wardley's Book")
st.title("Chat with Simon Wardley's Book")
st.sidebar.markdown("# Query Simon's book using AI")
st.sidebar.markdown("Developed by Mark Craddock](https://twitter.com/mcraddock)", unsafe_allow_html=True)
st.sidebar.markdown("Current Version: 1.1.0")
st.sidebar.markdown("Using gpt-3.5-turbo-16k-0613")
st.sidebar.markdown(st.session_state.session_id)
st.sidebar.markdown("Wardley Mapping is provided courtesy of Simon Wardley and licensed Creative Commons Attribution Share-Alike.")
st.sidebar.divider()

# Check if the user has provided an API key, otherwise default to the secret
user_openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", placeholder="sk-...", type="password")

    
# Get datastore
DATA_STORE_DIR = "data_store"

if "vector_store" not in st.session_state:
    if os.path.exists(DATA_STORE_DIR):
        st.session_state.vector_store = FAISS.load_local(
            DATA_STORE_DIR,
            OpenAIEmbeddings()
        )
    else:
        st.write(f"Missing files. Upload index.faiss and index.pkl files to {DATA_STORE_DIR} directory first")
    
    custom_system_template="""
        As a friendly and helpful assistant with expert knowledge in Wardley Mapping,
        Analyze the provided book on Wardley Mapping and offer insights and recommendations.
        Suggestions:
        Explain the analysis process for a Wardley Map
        Discuss the key insights derived from the book
        Provide recommendations based on the analysis
        Use the following pieces of context to answer the users question.
        If you don't know the answer, just say that "I don't know", don't try to make up an answer.
        Your primary objective is to help the user formulate excellent answers by utilizing the context about the book and 
        relevant details from your knowledge, along with insights from previous conversations.
        ----------------
        Reference Context and Knowledge from Similar Existing Services: {context}
        Previous Conversations: {chat_history}"""
    
    custom_user_template = "Question:'''{question}'''"
    
    prompt_messages = [
        SystemMessagePromptTemplate.from_template(custom_system_template),
        HumanMessagePromptTemplate.from_template(custom_user_template)
        ]
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    # If the user has provided an API key, use it
    # Swap out openai for promptlayer
    promptlayer.api_key = st.secrets["PROMPTLAYER"]
    openai = promptlayer.openai
    openai.api_key = user_openai_api_key

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key='answer')

if "llm" not in st.session_state:
    st.session_state.llm = PromptLayerChatOpenAI(
        model_name=MODEL,
        temperature=0,
        max_tokens=300,
        pl_tags=["bookchat", st.session_state.session_id],
    )  # Modify model_name if you have access to GPT-4

if "chain" not in st.session_state:
    st.session_state.chain = ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=st.session_state.vector_store.as_retriever(
            search_kwargs={
                "k": 3,
                #"score_threshold": .95,
                }
            ),
        chain_type="stuff",
        rephrase_question = True,
        return_source_documents=True,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if user_openai_api_key:
    if query := st.chat_input("What question do you have for the book?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
    
        with st.spinner():
            with st.chat_message("assistant"):
                response = st.session_state.chain(query)
                st.markdown(response['answer'])
                st.divider()
                
                source_documents = response['source_documents']
                for index, document in enumerate(source_documents):
                    if 'source' in document.metadata:
                        source_details = document.metadata['source']
                        cleaned_content = clean_text(document.page_content)
                        st.warning(f"Source {index + 1}: Page {document.metadata['page']}\n")
                        st.write(f"{cleaned_content}\n")
    
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
else:
    st.warning("Please enter your OpenAI API key", icon="⚠️")
