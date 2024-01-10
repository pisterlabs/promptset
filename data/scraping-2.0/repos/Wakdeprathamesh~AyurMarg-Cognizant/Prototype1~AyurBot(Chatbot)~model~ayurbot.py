from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder)
import streamlit as st
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pinecone
from utils import *
from openai import OpenAI
import os



st.set_page_config(page_title='Ayurveda Chatbot', page_icon=':herb:')
st.header('Ayurvedic chatbot for a healthy routine :herb:')
with st.sidebar:
    st.title('Ayurveda Chatbot')
    st.markdown('''
        ## About
        Personalised Ayurveda Chatbot built using classic ayurvedic books
        ''')
    st.write('Made with ❤️ by [Team Ayurmarg](https://www.youtube.com/watch?v=6_fYrOHm7QE&t=13s)')

load_dotenv("D:\\Computer_Programming\\Python\\ayurveda_model\\.env")

client = OpenAI(
    api_key = os.environ['OPENAI_API_KEY']
)

if 'responses' not in st.session_state:
    st.session_state['responses'] = ['How can I help you?']
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
    
llm = ChatOpenAI(model_name = "gpt-3.5-turbo")

if 'buffer_mmeory' not in st.session_state:
    st.session_state.buffer_mmeory = ConversationBufferWindowMemory(k = 3, return_messages = True)
    
system_msg_template = SystemMessagePromptTemplate.from_template(template = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say 'I dont know'""")

human_msg_template = HumanMessagePromptTemplate.from_template(template = "{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name = "history"), human_msg_template])


conversation = ConversationChain(memory = st.session_state.buffer_mmeory, prompt = prompt_template, llm = llm, verbose = False)

response_container = st.container()
text_container = st.container()

model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(
    api_key = "e3600f56-6371-4ac1-9f29-fa22ea086c42",
    environment =  "gcp-starter"
)

index = pinecone.Index('ayurveda-chatbot')


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k = 2, include_metadata = True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']



def query_refiner(conversation, query):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant providing information only on ayurveda related topics"},
        {"role": "user", "content": f"Context: {conversation}"},
        {"role": "assistant", "content": f"Query: {query}"}],

        temperature=0.5,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string



with text_container:
    query = st.text_input("How may I help you? ", key = "input")
    if query:
        with st.spinner("Generating the response..."):
            conversation_string = get_conversation_string()
            st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            context = find_match(refined_query)
            response = conversation.predict(input = f"Context: \n {context} \n\n Query: \n {query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)


with response_container:
    if st.session_state['responses']:

        
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key = str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user = True, key = str(i) + '_user')
            