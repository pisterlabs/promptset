import streamlit as st
from openai import OpenAI

client = OpenAI(api_key = st.secrets['OPENAI_API_KEY'])

if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": f"Go on, ask me questions."})
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input()
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        report = []
        for resp in client.chat.completions.create(
            model = "gpt-4",
            messages=[
                    {"role": "system", "content": "You are a detailed oriented and smart assistant."},
                    {"role": "user", "content": prompt},
                ],
            stream = True):
            if resp.choices[0].finish_reason == "stop":
                break
            # result = resp.choices[0].delta.content
            full_response += resp.choices[0].delta.content
            report.append(resp.choices[0].delta.content)
            result = "".join(report).strip()
            # result = result.replace("\n", "")    
            message_placeholder.markdown(f'{result}' + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
