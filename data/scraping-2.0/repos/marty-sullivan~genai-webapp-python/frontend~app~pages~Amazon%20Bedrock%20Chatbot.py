from langchain.schema import HumanMessage
from langserve.client import RemoteRunnable
from os import environ
import streamlit as st

backend_host = environ['BACKEND_HOST']
backend_port = environ['BACKEND_PORT']

claude_chain = RemoteRunnable(f'http://{backend_host}:{backend_port}/bedrock_claude')
llama_chain = RemoteRunnable(f'http://{backend_host}:{backend_port}/bedrock_llama')

st.title('🤖 Amazon Bedrock Chatbot')
st.caption('A simple chatbot that uses either the latest Anthropic Claude v2 LLM or Llama 2 Chat 70B LLM via Amazon Bedrock')

model_selection = st.selectbox('Select LLM', ['Claude', 'Llama'])

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

if prompt := st.chat_input('Type your question...'):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar='👤').write(prompt)

    if model_selection == 'Claude':
        response = claude_chain.stream(dict(
            messages=[
                HumanMessage(content=prompt),
            ],
        ))
    
    elif model_selection == 'Llama':
        response = llama_chain.stream(dict(
            messages=[
                HumanMessage(content=prompt),
            ],
        ))
    
    else:
        st.write('Error: Invalid model selection')

    with st.chat_message("assistant", avatar='🤖'):
        full_message = ''
        with st.empty():
            for message in response:
                full_message += message.content
                st.write(full_message)

    st.session_state.messages.append({"role": "assistant", "content": full_message})
