import os
import pickle
import random
import streamlit as st

from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GPT4All
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from streamlit.web.server import websocket_headers
from streamlit_chat import message



def get_chat_llm(model,openai_api_key):
        
    if model and openai_api_key:
        if model and model == "GPT-3.5":
            llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key,model_name='gpt-3.5-turbo')
        elif model and model == "GPT-4":
            llm = llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key,model_name='gpt-4')
    
    return llm    



def get_chain(vectorstore, model, openai_api_key):
    
    qa_chain = None
    llm = None
    
    if openai_api_key is None or not openai_api_key or openai_api_key == "":
        openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if model and openai_api_key:
        llm = get_chat_llm(model, openai_api_key)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if llm:
        qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory, qa_prompt=QA_PROMPT, condense_question_prompt=CONDENSE_QUESTION_PROMPT)

    return qa_chain



_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about employee benefits in health care policy documentation.
You are given the following extracted parts of a long document and a question. 
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about employee benefits or policy coverage, politely inform them that you are tuned to only answer questions pertaining to policy coverage.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""

QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])





# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []

st.set_page_config(initial_sidebar_state='collapsed')
# Uncomment if you want to get the key from the user
openai_key = st.sidebar.text_input("Enter your OpenAI API key", type="password") 
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
docs_source = st.sidebar.radio("Choose a store:", ("Employee Benefits", "DDL Docs"))
clear_button = st.sidebar.button("Clear Conversation", key="clear")


store = None 
if docs_source == "Employee Benefits":
    # Load the embeddings from the pickle file; change the location if needed
    with open("healthcareplandetails.pkl", "rb") as f:
        store = pickle.load(f)
elif docs_source == "DDL Docs":
    with open("faiss_ddl_doc_store.pkl", "rb") as f:
        store = pickle.load(f)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    memory.clear()


# if store and openai_key:
if store: 
    qa = get_chain(store, model_name, openai_key)


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
    if submit_button and user_input and qa:
        with st.spinner("Searching for the answer..."):
            with get_openai_callback() as cb:
                result = qa({"question": user_input})
                answer = result["answer"]

        st.session_state['total_tokens'].append(cb.total_tokens)
        answer = result["answer"]
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(answer)
        
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, logo='https://freesvg.org/img/1367934593.png', key=str(i) + '_user')
                message(st.session_state["generated"][i], logo='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQk6e8aarUy37BOHMTSk-TUcs4AyAy3pfAHL-F2K49KHNEbI0QUlqWJFEqXYQvlBdYMMJA&usqp=CAU', key=str(i))


            