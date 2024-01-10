import streamlit as st
import cohere

st.title("Masters of Scale")

# initialize the cohere client
co = cohere.Client(st.secrets["COHERE_KEY"]) # with trial API key


if "model" not in st.session_state:
    st.session_state["model"] = "command"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in co.chat(
            model='command',
            temperature=0.3,
            prompt_truncation='AUTO',
            stream=True,
            citation_quality='accurate',
            connectors=[{"id":"web-search","options":{"site":"https://mastersofscale.com"}}],
            documents=[],
            message=prompt,
        ):
            if hasattr(response, 'text'):
                print(response.text)
                full_response += response.text
            else:
                print("Received non-text response:", repr(response))
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

