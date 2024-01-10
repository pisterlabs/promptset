import openai
import streamlit as st
import requests

st.title("中医康养")

openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
prompt ='你是一个专业的中医师，需要给出专业的康养建议。'
if question := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt+question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
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
        except openai.error.APIError as e:
            message_placeholder.markdown(f"**Error:** {e}")
        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred: {e}")
            

        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})