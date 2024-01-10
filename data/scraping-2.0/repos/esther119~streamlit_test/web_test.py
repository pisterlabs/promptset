"""Python file to serve as the frontend"""
import streamlit as st
from PIL import Image

from streamlit_chat import message
import pinecone
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
import openai
import time


icon = Image.open('images/wbw.jpeg')

embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=st.secrets['openai_api_key'])


pinecone.init(
    api_key=st.secrets['pinecone_api_key'],
    environment='asia-southeast1-gcp-free'   
)
index_name =  'tim-urban-test'



def load_chain():
    """Logic for loading the chain you want to use should go here."""
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    return docsearch

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="waitbutwhy ai", page_icon=icon, layout="wide")

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.header("When I dream to write like wait but why")
open_pic = Image.open('images/balls.png')
open_pic_resize = open_pic.resize((700, 400))
st.image(open_pic_resize, caption='waitbutwhy.com')


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("Write a wait but why style blog about: ", "first principles", key="input")
    return input_text

user_input = get_text()

def similarity_search(user_input):
    docs = chain.similarity_search(user_input)
    return docs[0].page_content

def generate_images(user_input, openai_api_key):
    openai.api_key = openai_api_key
    PROMPT = f'Draw a stick man about {user_input}, black marker, simple, long stick body, short hand, big round face'
    response = openai.Image.create(
        prompt=PROMPT,
        n=1,
        size="256x256",
    )
    return response["data"][0]["url"]


def AI_response_messages(user_input, store, openai_api_key):
    template ='''
    Use the following pieces of context from waitbutwhy to answer the question at the end. 
    If you don't know the answer, just clarify that you are not sure, but this might be how Tim Urban thinks.
    '''
    engineered_user_input = f'Write a blog about {user_input} within 300 words like waitbutwhy using "you" in a casual language'
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    from langchain.chat_models import ChatOpenAI
    chat = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613', openai_api_key=openai_api_key)   

    print("engineered_user_input", engineered_user_input)
    messages = [
        SystemMessage(content=template+store),
        HumanMessage(content=engineered_user_input)
    ]
    response=chat(messages)

    return response

if user_input:
    if user_input:
        with st.spinner("We are thinking hard with Tim's brain..."):
            time.sleep(7)        
        store = similarity_search(user_input)
        response = AI_response_messages(user_input, store, st.secrets['openai_api_key'])
        image_url = generate_images(user_input, st.secrets['openai_api_key'])
        # image_url = "https://i1.sndcdn.com/avatars-000172456930-u1912p-t500x500.jpg"
        # response = "hello"
    # st.write("context search: ", store)    
    st.session_state.past.append(f'Write a blog about {user_input}')
    # st.session_state.generated.append([response, image_url])
    st.session_state.generated.append([response.content, image_url])


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        text= st.session_state["generated"][i][0]
        image_url = st.session_state["generated"][i][1]
        image = f'<img width="256" height="256" src="{image_url}"/>'
        message(text, key=str(i))
        message(image, allow_html=True)
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style = 'micah')