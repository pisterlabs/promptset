import openai
import streamlit as st
from llama_index import Document, ServiceContext, StorageContext, VectorStoreIndex, load_index_from_storage
# from llama_index.llms import OpenAI
from llama_hub.remote_depth import RemoteDepthReader
import tldextract

DEFAULT_URL = 'https://www.xomnia.com'

st.set_page_config(page_title='Chat', page_icon=f'assets/brain.png', layout="centered", initial_sidebar_state="auto", menu_items=None)

st.session_state.update(st.session_state)

default_title = st.title(f'Website Chat')

avatars = {
    'user': 'assets/programmer.png',
    'assistant': f'assets/brain.png'
}

def load_data():
    with st.spinner(text="Loading indexed documents..."):
        # RemoteDepthReader = download_loader("RemoteDepthReader")
        loader = RemoteDepthReader(depth=depth, domain_lock=True)
        documents = loader.load_data(url=url)
        st.success(f'{len(documents)} documents loaded')
        # service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt=f"You are an expert on the {name} website and your job is to answer technical questions. Assume that all questions are related to {name}. Keep your answers technical and based on facts – do not hallucinate features."))
        # index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        return index

with st.sidebar:
    url = st.text_input('Enter a URL', DEFAULT_URL)
    parsed_url = tldextract.extract(url)
    name = parsed_url.domain
    depth = st.radio('Depth', [0, 1], horizontal=True, index=0)

    if "index" not in st.session_state:
        if st.button('Load website'):
            st.session_state.index = load_data()
            st.success('Website loaded')


if 'index' in st.session_state:
    default_title.title(f'{name.title()} Chat')

    # Initialize the chat messages history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": f"Ask me a question about {url}!"}
        ]

    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
            st.session_state.chat_engine = st.session_state.index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"], avatar=avatars[message["role"]]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar=avatars["assistant"]):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history