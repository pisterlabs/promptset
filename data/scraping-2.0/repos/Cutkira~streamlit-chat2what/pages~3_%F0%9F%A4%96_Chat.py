import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Set chat page
st.set_page_config(page_title="Chat", layout="wide")
st.title("Chat 🤖")

# Initialize ChatGPT Object
chat = None

if "LANGUAGE" not in st.session_state:
    st.session_state["LANGUAGE"] = "简体中文"

# set button string
if st.session_state["LANGUAGE"] == "简体中文":
    openai_error_string = "哎呀！ 请在Setting标签页中设置你的OpenAI API Key。"
    chat_input_string = "聊点嘛呢..."
elif st.session_state["LANGUAGE"] == "English":
    openai_error_string = "Oops! Please set your OpenAI API Key in the settings page."
    chat_input_string = "Type something..."

# switch page
if "switch_page" not in st.session_state:
    st.session_state["switch_page"] = "Chat"

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""
elif st.session_state["OPENAI_API_KEY"] != "":
    chat = ChatOpenAI(openai_api_key=st.session_state["OPENAI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Main
if chat:
    st.header("Chat with GPT")
    # create chat container, display history of chatting
    for message in st.session_state["messages"]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    prompt = st.chat_input(placeholder=chat_input_string)
    if prompt:
        # process prompt, user message
        st.session_state["messages"].append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        # get response from OpenAI, its type is AImessage now.
        ai_message = chat([HumanMessage(content=prompt)])
        st.session_state["messages"].append(ai_message)
        with st.chat_message("assistant"):
            st.markdown(ai_message.content)
else:
    with st.container():
            st.error(openai_error_string)

st.session_state["switch_page"] = "Chat"
