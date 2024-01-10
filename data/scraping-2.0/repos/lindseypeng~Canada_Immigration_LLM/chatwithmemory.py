import os
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3



with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"



#openai.api_key = openai_api_key


##inference api vs inference endpoint hugging face
#https://github.com/langchain-ai/langchain/issues/3275
## load keys
#openai_api_key = os.environ.get('OPENAI_API_KEY')
#huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']
##load models
###openai



###hugging face

#repo_id = "tiiuae/falcon-7b-instruct"
#repo_id = "TheBloke/Wizard-Vicuna-30B-Superhot-8K-fp16"
#repo_id = "google/flan-t5-base"
#llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     #repo_id=repo_id, 
                     #model_kwargs={"temperature":0.5, "max_new_tokens":250})
## load context embedding
#persist_directory = '/Users/linqianpeng/Documents/immigration/docs/chroma'


st.title('question answering canadian immigration')

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot for Canadian PR powered by OpenAI LLM")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("what is your question?"):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    # Display user message in chat message container
    openai.api_key = openai_api_key
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
# question= st.text_input('Enter your question here')
    llm_name= "gpt-3.5-turbo"
    llm = ChatOpenAI(openai_api_key=openai_api_key,model_name = llm_name,temperature=0)
    persist_directory = './docs/chroma'
    embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    ## conversation AI with memory for chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory)

    ## conversation AI with no memory, return reference
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),return_source_documents=True,
        chain_type="stuff"
)
    result = qa_chain({"query": prompt})
    response = result['result']
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
# if st.button('Get Answer'):
#     if question:
#         #result = qa({"question": question})
#         result =  qa_chain({"query": question})
#         #response=result['answer']
#         response=result['result']
#         st.write(question)
#         st.write(response)
#         st.write()
#         for i in result['source_documents']:
#             st.write('References:')
#             st.write()
#             st.write(i.page_content)
#             st.write(i.metadata)
#             st.write()

#     else:
#         st.write('no question found')

    
    