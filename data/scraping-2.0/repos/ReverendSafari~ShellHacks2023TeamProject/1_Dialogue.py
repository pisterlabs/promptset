import streamlit as st
import openai
import data
import conversor as convo
import sqlite3 as db
import static


#static.ERASE_DATA()

data.init_db()

if (not data.analysis.is_null()):
    data.analysis.nullify()

    # Organization for the UI
col1, col2 = st.columns(2)

with col1:
    tgLang = st.selectbox('Target Language', convo.system.LANGS)

with col2:
    data.current.ctype = st.selectbox('Dialog Type', ['Conversation', 'Advice and corrections'])

static.general()


    # Show the chat feature only if the user is logged in
if static.st.session_state.is_logged_in:
    if (static.current.user is not None):
        st.title("LangGPT - " + static.current.user.sysname)
        user_input = st.text_input('Enter a sentence')
    else:
        st.title("LangGPT")
        user_input = False
    
    if  static.current.user and data.current.ctype and st.button("Send") and user_input:
        # Simulate user's message
        if (data.current.ctype == 'Conversation'):
            static.current.user._converse(user_input, tgLang)
        else:
            static.current.user._feedback(user_input, tgLang)

        # Display Chat History
        st.write("Chat History")

        if (data.current.ctype == 'Conversation'):
            history = static.current.user.conversations[tgLang].dialog_list
        else:
            history = static.current.user.langchecks[tgLang].dialog_list

        for i in range (1, len(history)):
            if history[i]["role"] == "user":
                st.write(static.current.user.name + f": {history[i]['content']}")
            else:
                st.write(static.current.user.sysname + f": {history[i]['content']}")   

else:
    st.warning("Please log in to access the chat feature")
    user_input = False
