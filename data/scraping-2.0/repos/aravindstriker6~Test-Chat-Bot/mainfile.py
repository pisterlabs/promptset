from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
load_dotenv()
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from kelly_bot_test import *

st.subheader("Test Chat Bot")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi , hope you had a good day. May I know how can I assist you today"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []
if 'buffer_memory' not in st.session_state:
    st.session_state['buffer_memory'] = ConversationBufferWindowMemory(k=5, return_messages=True)

llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", openai_api_key=os.getenv("OPEN_API_KEY"))

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])


conversation = ConversationChain(memory=st.session_state['buffer_memory'],prompt=prompt_template, llm=llm, verbose=True)



response_container = st.container()
textcontainer = st.container()

uploads_folder = r"C:\Users\asrinivasan\chatbot\Scripts\uploads"
os.makedirs(uploads_folder, exist_ok=True)

file_path = None

with textcontainer:
    query = st.text_input("Query: ", key="input")
    
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            # print(context)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

print(st.session_state)
    
