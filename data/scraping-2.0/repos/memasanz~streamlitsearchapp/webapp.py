import json
import openai
import streamlit as st

from streamlit_chat import message
from streamlit_option_menu import option_menu
from credentials import * 
from cog_search import *

def update_creds():
    with open('credentials.py') as f:
        l = list(f)

    for attribute, value in creds.items():
        with open('credentials.py', 'w') as output:
            for line in l:
                    if line.startswith(attribute):
                        print('found attribute: ' + attribute + ' = "' +  value + '"\n')
                        print('about to write: ' + attribute + ' = "' +  value + '"\n')
                        output.write( attribute + ' = "' +  value + '"\n')
                    else:
                        output.write(line)
    f.close()
    output.close()

#reads from credentials.pyand puts values of creds into session_state
def set_creds():
    for attribute, value in creds.items():
        if attribute not in st.session_state:
            st.session_state[attribute] = value

#print session_state message for chat like experience
def print_messages():
        for i in range (len(st.session_state.messages) -1, -1, -1):
            msg = st.session_state.messages[i]
            if msg is not None:
                if msg["role"] == "user":
                    message(msg["content"], is_user=True, key = str(i) + "user", avatar_style = "initials", seed = "👤")
                else:
                    if msg["role"] == "assistant":
                        print(msg["content"])
                        if msg["content"] == "I don't know" or msg["content"] == "I don't know." or msg['content'] == "Sorry, I don't know the answer to that question. Please try rephrasing your question.":
                            message(msg["content"], is_user=False, key = str(i) + "system", avatar_style="initials", seed = "😕")
                        elif msg['content'] == "How can I help you?":
                            message(msg["content"], is_user=False, key = str(i) + "system", avatar_style="initials", seed = "🙂")
                        else:
                            message(msg["content"], is_user=False, key = str(i) + "system", avatar_style="initials", seed = "😉")

#clear session_state messages - this is on the settings page
def settings_form():
    reset = st.checkbox('Reset Messages')
    if reset:
        st.write('Sure thing!')
        st.session_state.messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
        st.session_state.messages.append({"role": "assistant", "content": "How can I help you?"}) 
        print("complteted reset")


#display home

 

#main screen
if __name__ == "__main__":
        
    if "messages" not in st.session_state:
        print("messages not in session state")
        st.session_state["messages"] = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
        st.session_state.messages.append({"role": "assistant", "content": "How can I help you?"})

    with st.sidebar:
        set_creds()
        menu_index = ['Home', 'Settings', 'Upload file', 'Chat']
        menu_icons = ['house', 'gear', 'cloud-upload',  'chat']

        selected = option_menu("Main Menu",menu_index, icons=menu_icons, menu_icon="cast", default_index=1)

    if selected == 'Home':
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"

        st.title("Welcome to the AI Assistant")
        with st.form("chat_input", clear_on_submit=True):
            a, b = st.columns([4, 1])
            user_input = a.text_input(
                label="Your message:",
                placeholder="What would you like to say?",
                label_visibility="collapsed",
            )
            b.form_submit_button("Send")


        openai.api_type = "azure"
        openai.api_base = creds['AZURE_OPENAI_ENDPOINT']
        openai.api_version = "2023-03-15-preview"
        openai.api_key = creds['AZURE_OPENAI_KEY']
            

        if user_input and AZURE_OPENAI_KEY:
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                messages = st.session_state.messages,
                temperature=0.0,
                max_tokens=200,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            print(response)
            if response.choices[0].message.content != None:
                st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})

        print_messages()

    elif selected == 'Upload file':
        st.title('Upload file')
    elif selected == 'Chat':
        st.title('Chat')
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
        st.title("Cognitive Search & Azure OpenAI")
        with st.form("chat_input", clear_on_submit=True):
            a, b = st.columns([4, 1])
            user_input_ondata = a.text_input(
                label="Your message:",
                placeholder="What would you like to ask?",
                label_visibility="collapsed",
            )
            b.form_submit_button("Send")
            
        if user_input_ondata and AZURE_OPENAI_KEY:
            question = user_input_ondata
            st.session_state.messages.append({"role": "user", "content": question})
            arg = OpenAIHelper(creds['COG_SEARCH_INDEX'])
            response = arg.get_Answer(user_input_ondata)
            st.session_state.messages.append({"role": "assistant", "content": response})

        print_messages()

    elif selected == 'Settings':
        settings_form()
        
        with st.form("my_form"):
            st.write("Configuration Settings")
        

            azure_openai_endpoint = st.text_input("azure_openai_endpoint", creds["AZURE_OPENAI_ENDPOINT"])

            azure_openai_key = st.text_input("azure_openai_key", creds["AZURE_OPENAI_KEY"], type = "password")
            txt_davinci = st.text_input("txt davinici", creds["TEXT_DAVINCI"])

            cog_search_resource = st.text_input("Cog Search Resource",creds["COG_SEARCH_RESOURCE"])
            cog_search_index = st.text_input("Cog Search Index", creds["COG_SEARCH_INDEX"])
            cog_service_key   = st.text_input("Cog Search Key", creds["COG_SEARCH_KEY"], type = "password")

            storage_connection_string = st.text_input("Storage Connection String", creds["STORAGE_CONNECTION_STRING"], type="password")
            storage_account = st.text_input("Storage Account", creds["STORAGE_ACCOUNT"])
            storage_container = st.text_input("Storage Container", creds["STORAGE_CONTAINER"])
            storage_key = st.text_input("Storage Key", creds["STORAGE_KEY"], type = "password")

            submitted = st.form_submit_button("Submit")

            #don't use this to update the search index.
            if submitted:
                creds["AZURE_OPENAI_ENDPOINT"] = azure_openai_endpoint
                creds["AZURE_OPENAI_KEY"] = azure_openai_key
                creds["TEXT_DAVINCI"] = txt_davinci
                creds["COG_SEARCH_RESOURCE"] = cog_search_resource
                #creds["COG_SEARCH_INDEX"] = cog_search_index
                creds["COG_SEARCH_KEY"] = cog_service_key
                creds["STORAGE_CONNECTION_STRING"] = storage_connection_string
                creds["STORAGE_ACCOUNT"] = storage_account
                creds["STORAGE_CONTAINER"] = storage_container
                creds["STORAGE_KEY"] = storage_key
                set_creds()
                # update_creds()
                st.write("Settings updated")
        
        with st.form("create index"):
            
            st.write("Create Index")
            create_index = st.form_submit_button("SubmitCreateIndex")

        if create_index:
            cogSearch = CogSearchHelper(index = creds["COG_SEARCH_INDEX"])
            response, success = cogSearch.create_datasource()
            if success:
                st.write("Data source created")
                response, success = cogSearch.create_skillset()
                if success:
                    st.write("Skillset created")
                    response, success = cogSearch.create_index()
                    if success:
                        st.write("Index created")
                        with st.spinner(text="In progress..."):
                            response, success = cogSearch.create_indexer()
                            if success:
                                st.write("Running indexer")