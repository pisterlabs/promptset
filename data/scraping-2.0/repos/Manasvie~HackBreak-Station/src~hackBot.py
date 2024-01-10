import openai
import streamlit as st
import json
import requests
from streamlit_lottie import st_lottie

def main():
    url = requests.get("https://lottie.host/8dc29224-f6b2-4d7d-a998-090bda0f517b/RwbR4PLOXr.json")

    url_json=dict()

    if url.status_code == 200:
        url_json=url.json()
    else:
        print('error')
    # Use HTML to embed the GIF in the sidebar
    st_lottie(url_json)

    st.title("Hack-Bot")
    st.subheader("Take a break 🥤, have a chat!")

    openai.api_key = st.secrets["OPENAI_API_KEY"]

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    #initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages =[]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    prompt = st.chat_input("What is up?")
    if prompt:
        # Display user mesage in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt}) 

        with st.chat_message("assistant"):
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
                full_response += response.choices[0].delta.get("content", "") # type: ignore
                message_placeholder.markdown(full_response + "")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})       

if __name__ == "__main__":
    main()
