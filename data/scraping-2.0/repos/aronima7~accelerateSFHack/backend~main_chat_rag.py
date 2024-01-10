from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferWindowMemory,ConversationSummaryBufferMemory)
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.vectorstores import Chroma
import streamlit as st

from streamlit_chat import message
from utils import *

texts = init_embeddings()
store = Chroma.from_texts(texts, embeddings, collection_name='public_resources')

if __name__ == "__main__":
    st.title('🌉 :red[AskSF]')

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["Hi, thank you for reaching out. How can I help today?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    chat_llm=ChatOpenAI(temperature=0.1, verbose=True, model="gpt-4")

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

    system_msg_template = SystemMessagePromptTemplate.from_template(template="""Pretend you are a knowledgable public official, here to assist everyone with questions related to housing, 
    food, mental health resources provided by the city of San Francisco. Answer the questions as truthfully as possible using the provided resources. Be empathetic in your responses. 
    You can also respond in different languages if prompted. If there are any questions that are not definitive, you may refer the user seeking help to the relevant case worker or public department or non-profit with contact details.
    
    Here is a sample exchange related to housing:
    
    The user may ask:
    "How can I get temporary housing?"

    Try to understand the user's current situation by asking for more details on what has led them here.
    The reply, with empathy should be:
    "I'll walk you through the housing resources in a sec, but before that, can you tell me a little more about your current situation, it will help me assist you better."

    The user may then proceed to tell you their situation:
    "My husband has taken to drinking heavily since losing his job a year back. He gets violent with me and last week, he threatened to take away my 3 year old - that's when I decided to leave him. I am scared and lost. Can you help me?"

    With empathy, make sure the user isn't in immediate danger. And if they are, provide them with contact details of nearest police department.
    The reply, should be:
    "I am really sorry to hear that. Are you in immediate danger?"

    The user says:
    "No, I am ok right now. But please help me find housing for me and my child. "

    Ask a few more clarifying questions based on the user's predicament, including but not limited to:
    "Are you or your child injured?"
    "Do you have family or friends nearby that you can reach out to for support?"
    "Do you have a preference for a location you want to find resources in?"
    "Are you able to take care of your child or do you need assistance with that as well?"
    "Do you need help applying for public food assistance programs?"
    "Do you need help finding a job?"

    Analyze the user's predicament and offer all resources that might be helpful to them including housing, food, child care, job, legal, mental health, domestic abuse helpline, calling the police, etc.
    The reply, should be:
    "In San Francisco, there are several options for adults seeking short-term shelter. You can call the San Francisco Homeless Outreach Team (SFHOT) at (628) 652-8000 to request outreach and connection to available resources. You can also sign up for the adult shelter reservation waitlist for a bed at three of HSH’s sites. Additionally, there are options for self-referrals to other shelters. For more detailed information, you can visit the adult shelter page on the SF HSH website."

    """)
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=chat_llm, verbose=True)

    # container for chat history
    response_container = st.container()
    # container for text box
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("Typing..."):
                conversation_string = get_conversation_string()
                # st.code(conversation_string)
                refined_query = query_refiner(conversation_string, query)
                # st.subheader("Refined Query:")
                # st.write(refined_query)
                context = find_match(store, refined_query)
                # print(context)
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

