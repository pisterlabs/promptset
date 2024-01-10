import streamlit as st
from openai import OpenAI

import time

client = OpenAI(api_key=st.secrets.openai.api_key_general)
assistantid = st.secrets.openai.assistant_key_threadselector
userprompt = "Tell me something about travel"

def get_response_from_message_list(vMessageList):
    data = vMessageList.data
    print(data)
    data_response = data[0]
    content = data_response.content[0]
    text = content.text
    value = text.value
    return value

def get_selected_thread_messages(vThreadId):
    messagelist = client.beta.threads.messages.list(
        thread_id=vThreadId
    )
    return messagelist

def get_file(vFileIds):
    file = client.files.content(
        file_id=vFileIds
    )
    return file
    

def thread_selector(vUserPrompt):

    # Create thread for user prompt
    thread_main = client.beta.threads.create()
    thread_main_id = thread_main.id

    #Add the user prompt 
    message = client.beta.threads.messages.create(
        thread_id=thread_main_id,
        role="user",
        content=vUserPrompt
    )
    messageid = message.id

    #run assistant to select the thread
    run = client.beta.threads.runs.create(
        thread_id=thread_main_id,
        assistant_id=assistantid
    )

    #check run status
    while run.status=="in_progress" or run.status=="queued":
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            run_id=run.id,
            thread_id=thread_main_id
        )

        if run.status=="completed":
            messagelist = client.beta.threads.messages.list(
                thread_id=thread_main_id
            )

            messageresponse = get_response_from_message_list(messagelist)
            return messageresponse


msgresponse = thread_selector(userprompt)
selected_thread_messages = get_selected_thread_messages(msgresponse)
print(msgresponse)
print(selected_thread_messages)
