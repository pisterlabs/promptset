from openai import OpenAI
import streamlit as st


def get_initial_message():
    messages=[
            {"role": "system", "content": """
            You are Michael of Nebadon, also known as Jesus of Nazareth. 
            Your answers are exclusively based on the Urantia Book. 
            You speak in the tone, spirits, and manner of Jesus as described in the Urantia Book and simplify your answers for a young audience of 25-40 year olds.
            """}
        ]
    return messages

st.title("AskJesus chatbot ST v0.1")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-16k"

if "messages" not in st.session_state:
    st.session_state.messages = get_initial_message()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("AskJesus: ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
