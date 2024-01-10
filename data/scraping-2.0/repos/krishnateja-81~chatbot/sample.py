import openai 
import streamlit as st

st.set_page_config(page_title="KH-CHATBOT", page_icon='screenshot (2).png')
openai.api_key = ""
st.title("CHATGPT INTEGRATED CHATBOT", anchor="str")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"],avatar="🤖"):
        st.markdown(message["content"])
try:
    if prompt := st.chat_input("send a message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user",avatar="👨‍💻"):
            st.markdown(prompt)
        
        with st.chat_message("assistant",avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                stream=True,
                ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
except:
    message_placeholder.markdown("SERVERS ARE BUSY")
