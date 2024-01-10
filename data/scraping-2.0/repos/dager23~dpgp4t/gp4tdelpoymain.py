import streamlit as st

from openai import OpenAI

client = OpenAI(api_key=st.secrets["APIKEY"],organization=st.secrets["ORG"])


def response(text):
    response = client.chat.completions.create( model="gpt-4", messages=[{"role": "user", "content":text }] ) 
    return response.choices[0].message.content

if "messages" not in st.session_state:
                st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    #text=st.text_input('Enter your query here')

# React to user input
if text:= st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(text)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": text})

    response = f"Bot: {response(text)}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

