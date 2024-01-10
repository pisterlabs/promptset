import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from cg_utils import *


# Get text-to-text FMs
t2t_fms = get_t2t_fms(fm_vendors)


def fm_chat(modelid:str):
    """Start chat with specific FM"""
    msgs = StreamlitChatMessageHistory(key=f"{modelid}_key")
    memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
    if len(msgs.messages) == 0:
       msgs.add_ai_message("How can I help you?")
    template = """You are an AI chatbot having a conversation with a human.

    {history}
    Human: {human_input}
    AI: """
    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
    fm = get_fm(modelid)
    fm_chain = LLMChain(llm=fm, prompt=prompt, memory=memory)
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
    prompt_chat = st.chat_input(placeholder="Start a conversation!", key="prompt_key")
    if prompt_chat:
        st.chat_message("human").write(prompt_chat)
        response = fm_chain.run(prompt_chat)
        st.chat_message("ai").write(response)


def main():
    """Main function for Chat"""
    st.set_page_config(page_title="Chat with an FM", layout="wide")
    css = '''
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
            }
        </style>
    '''
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with Amazon Bedrock FMs")
    st.markdown("Select a foundation model and start chatting! Refer the [Demo Overview](Solutions%20Overview#fm-chat) for a description of the solution.")
    chat_fm = st.selectbox('Select Foundation Model', t2t_fms, key="chat_fm_key")
    if chat_fm:
        fm_chat(st.session_state.chat_fm_key)


# Main  
if __name__ == "__main__":
    main()