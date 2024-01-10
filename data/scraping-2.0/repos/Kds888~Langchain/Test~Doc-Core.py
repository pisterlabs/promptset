import os 
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone
from langchain import OpenAI

pinecone.init(api_key="YOUR API KEY HERE",environment="asia-southeast1-gcp-free") 

# The only reason we have api's keys have been there is because of strem lit other wise we could make use of the 
# .env varibales

index_name = 'doc-chat' 

def run_llm(query,history):
    embeddings = OpenAIEmbeddings(openai_api_key='sk-YOUR API KEY HERE')
    docsearch = Pinecone.from_existing_index(index_name=index_name,embedding=embeddings) 
    chat= ChatOpenAI(openai_api_key='sk-YOUR API KEY HERE',verbose=True,temperature=0)
    #qa = RetrievalQA.from_chain_type(llm=chat,chain_type='stuff',retriever=docsearch.as_retriever()) # here we need to say that 
    # return_source_documents=True , it will return us the document from which the answer was generated. and we can extract the link from
    # this document
    # We are not making use of the retrievalqa as it won't have the history embedded in it.
    qa = ConversationalRetrievalChain.from_llm(llm=chat,retriever=docsearch.as_retriever())
    return qa({'question':query,"chat_history":history})   

# ans=run_llm(query="How to get enormous wealth?") 
# # We can see that we jus need to send the queries and we will get the responses from the chat 
# print(ans)


################# FRONTEND ####################################################################################

import streamlit as st

# We have sesssion state defined already in the streamlit which basically means that we can save some of the responses in it

import time 
# In order ot run the stream lit we have to right the command in the CMD streamlit run C:\Users\karan\Test\Doc-Core.py
from streamlit_chat import message
st.header("DOCUMENT CHAT") 
prompt = st.text_input("PROMPT",placeholder="Enter your prompt here ")

# basically at the start of the session we are making it 

if 'user_prompt_history' not in st.session_state:
    st.session_state['user_prompt_history']=[]
if 'chat_answers_history' not in st.session_state:
    st.session_state['chat_answers_history']=[] 
# we are creating these dictionruy values so  that we can make changes to it in the session_state in the stream lit loop 
# and to basically display the contents on the screen 

if 'chat_history' not in st.session_state:
    st.session_state['chat_history']=[]
    # it will have both the question asked by the user and the answer responded by the llm

if prompt:
    with st.spinner("Generating Response ....."):
        generated_repsonse = run_llm(query=prompt,history=st.session_state['chat_history'])
        # we can attach the links for each context in the meta deta so that we can extract the link as well as each response will 
        # have a meta data 
    st.session_state['user_prompt_history'].append(prompt)
    st.session_state['chat_answers_history'].append(generated_repsonse['answer']) # it will keep the memory of the responses that 
    # were given to us by the user.
    st.session_state['chat_history'].append((prompt,generated_repsonse['answer']))
if st.session_state['chat_answers_history']:
    for g_response,user_query in zip(st.session_state['chat_answers_history'],st.session_state['user_prompt_history']):
        message(user_query,is_user=True)
        message(g_response)
# right now we have not implemented memeory in our application therefore we need to have memory added to our data 




