import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)
sys.path.append(parent_dir)


import openai
import streamlit as st

from db.utils import SaveFileToDisk, VectorDB, ProcessDocument

st.title("ProtoGPT")

openai.api_key = os.getenv("OPENAI_API_KEY")

uploaded_file = st.sidebar.file_uploader(
    "Upload a protocol", 
    type=["pdf"]
    )

vector_db = VectorDB()

if uploaded_file:
    file_path = SaveFileToDisk(uploaded_file, vector_db).file_path
    st.write(file_path)
    file_name = os.path.basename(file_path)
    st.write(f"File name: {file_name}")

    if not vector_db.document_exists(file_name):
        documents = ProcessDocument(file_path).load_and_chunk()
        vector_db.add_documents(documents)
        vector_db.add_processed_document(file_name) 
    else:
        st.write(f"The document {file_name} has already been processed.")

select_file = st.sidebar.selectbox(
    "Select a protocol",
    vector_db.get_document_names(),
    index=0
)
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = 'gpt-3.5-turbo'

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    
    if select_file:
        st.write(select_file)
        context = vector_db.get_context(prompt, select_file)
    else:
        context = vector_db.get_context(prompt)
    
    final_prompt = context + "\n\n" + prompt 
    st.write(context)
    st.session_state.messages.append({"role": "user", "content": final_prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

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
            temperature=1.0,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})