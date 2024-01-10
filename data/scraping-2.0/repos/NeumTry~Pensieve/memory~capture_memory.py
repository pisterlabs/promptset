import streamlit as st
from typing import List
from documents.neumai_utils import createPipeline, triggerPipeline
from documents.upload_util import uploadFile
from memory.memory_utils import store_messages, search_chat_memory, search_document_memory, generate_and_store_summary, create_user
import openai
import uuid

openai.api_key=st.secrets["OpenAI_Key"]
system_message = {"role": "system", "content": "You are an assisstant that helps people do brain dumps about topics. Your job is to ask questions and clarify topics that are discussed. Don't be judgemental or pushy. Simply follow the conversation. This is some additional context you might use: {}"}

def capture_memory():
    if "capture_messages" not in st.session_state:
        st.session_state["capture_messages"] = []
        st.session_state["capture_messages"].append({"role": "assistant", "content": "What would you like to talk about?"})

    if "files_already_uploaded" not in st.session_state:
        st.session_state["files_already_uploaded"] = []
    
    if "user_created" not in st.session_state:
        st.session_state['user_created'] = False

    # Create weaviate class
    if st.session_state['user_created'] == False:
        create_user(user=st.session_state['capture_user'])
        st.session_state['user_created'] = True
    
    user = st.session_state['capture_user']
    session = st.session_state['capture_session']
    
    st.header("Capture Memory", divider=True)

    for msg in st.session_state.capture_messages:
        if(msg["role"] != "system"):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if prompt := st.chat_input():
        message_to_store = []
        history_retrieved = search_chat_memory(query=prompt,user=user, session=session)
        if len(history_retrieved) > 0:
            history = "\n".join(f"{message['role']}: {message['content']}" for message in history_retrieved)
        else: 
            history_retrieved = []
            history = "No history available"
        document_history_retrieved = search_document_memory(query=prompt,user=user, session=session)
        if len(document_history_retrieved) > 0:
            document_history = "\n".join(f"Document: {result['text']}" for result in document_history_retrieved)
        else: 
            document_history_retrieved = []
            document_history = "No history available"
        message_to_store.append({"role": "user", "content": prompt})
        st.session_state.capture_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # prepare messages payload to send. We will add the retrieved messages + the last couple messages in the convo
        local_chat_history = st.session_state.capture_messages[-5:]
        system_message_temp = system_message
        system_message_temp['content'] = system_message_temp['content'].format(document_history)
        system_message['content'].format(document_history)
        history_to_send = [system_message_temp] + [{"role": message['role'], "content":message['content']} for message in history_retrieved] + local_chat_history
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=history_to_send, stream=True):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.capture_messages.append({"role": "assistant", "content": full_response})
        message_to_store.append({"role": "assistant", "content": full_response})
        if st.session_state['debug']:
            st.text_area("Document History",document_history)
            st.text_area("Chat History",history)
            st.text_area("System prompt",system_message_temp)
        store_messages(messages=message_to_store, user=user, session=session, message_id=uuid.uuid4())
        message_to_store = []
    
    if st.button("Save memory", use_container_width=True):
        generate_and_store_summary(messages=st.session_state.capture_messages, user=user, session=session)
    
    # Handle file upload into the session memory
    uploaded_files = st.file_uploader(accept_multiple_files=True, label="Add files to your memory")
    new_files_to_upload = [file for file in uploaded_files if file.name not in st.session_state["files_already_uploaded"]]
    if(len(new_files_to_upload) > 0):
        # Upload files
        for uploaded_file in new_files_to_upload:
            bytes_data = uploaded_file.read()
            upload_status = uploadFile(file_bytes=bytes_data, file_name=uploaded_file.name, user=user, session=session)
            st.session_state["files_already_uploaded"].append(uploaded_file.name)
        
        # Trigger Neum Pipeline to run
        # First check if pipeline has been created
        if upload_status == 200:
            if st.session_state["neumai_pipeline"] == "":
                st.session_state["neumai_pipeline"] = createPipeline(user=user, session=session)
                # This will automatically trigger the pipeline
            else:
                # Trigger pipeline with pipeline id. This pipeline is configured with the user and session.
                triggerPipeline(st.session_state["neumai_pipeline"] )
        
            st.chat_message("assistant").write("Thanks for uploading a file. I have added it your memory.")
        else:
            st.chat_message("assistant").write("File upload failed. Try again later.")

        