import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

def text():

    apiKey = "AIzaSyAXkkcrrUBjPEgj93tZ9azy7zcS1wI1jUA"
    msgs = StreamlitChatMessageHistory(key="special_app_key")

    memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")
    template = """You are an AI chatbot having a conversation with a human.

    {history}
    Human: {human_input}
    AI: """
    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
    llm_chain = LLMChain( llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=apiKey), prompt=prompt, memory = memory)

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Say something")

    if prompt:
        with st.chat_message("user").markdown(prompt):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )
        for chunk in llm_chain.stream(prompt):
            text_output = chunk.get("text", "")

        with st.chat_message("assistant").markdown(text_output):
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": text_output
                }
            )





