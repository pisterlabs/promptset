
from openai import OpenAI
import streamlit as st
import os
os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH')}:/root/gpt_projects/ABoringKnowledgeManagementSystem/"
from DocumentIndexing.Elastic import search_engine
from DocumentIndexing.Embedding.embedding_local import embeddings_multilingual as embed

st.title(r"$\textsf{\tiny Knowing Bot}$")
with st.sidebar:
    rag = st.toggle("Using RAG", True)
    send_full_history = st.toggle("Send full chat history", False)
    dt_option = st.selectbox(
        'What type of document do you want to search?',
        ['research_paper', 'research_book', 'personal_document', 'others'])
    lang_option = st.selectbox(
            'What language do you want to search?',
        ['en','fr','cn'])
    title = st.text_input('Title', placeholder = 'Enter the title of the document') 
    author = st.text_input('Author',placeholder= 'Enter the author of the document')
    subject = st.text_input('Subject',placeholder= 'Enter the subject of the document')

system_message = {'role':"system","content":"""
                        You are a chatbot to help users to search documents,summerize documents and answer questions about documents.
                        Be precise and do not hallucinate."""}

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.session_state["openai_model"] = "gpt-4-1106-preview"

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Choose messages to send based on the toggle
        if send_full_history:
            messages_to_send = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
        else:
            messages_to_send = [system_message, {"role": "user", "content": prompt}]

        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=messages_to_send,
            stream=True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        if rag:
            new_serch_engine = search_engine.SearchEngine()
            print(prompt)
            embedded_query = embed(prompt)
            search_results = new_serch_engine.vector_search(index_name='research_paper', query_vector=embedded_query)
            print(search_results)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


