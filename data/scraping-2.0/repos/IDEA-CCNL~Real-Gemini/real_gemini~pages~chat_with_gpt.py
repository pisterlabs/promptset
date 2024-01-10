import streamlit as st
from openai import OpenAI
import os 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

st.title("ChatGPT-like")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=OPENAI_API_KEY)

# Set a default model
with st.sidebar:
    with st.form('参数配置'):
        model = st.selectbox('选择模型版本',['gpt-4', 'gpt-4 turbo', 'gpt-3.5-turbo'],index=0)
        st.form_submit_button('提交配置')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("输入你的问题"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        responses = client.chat.completions.create(model=model,
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,)
        for response in responses:
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})