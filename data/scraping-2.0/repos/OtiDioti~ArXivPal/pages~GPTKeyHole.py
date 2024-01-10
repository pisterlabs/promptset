from openai import OpenAI
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from st_pages import hide_pages # needed to hide pages
#%% 
hide_pages(st.session_state['pages_to_hide'])
#%%
def IsKeyValid(key):
    client = OpenAI(api_key = key)
    try:
        response = client.chat.completions.create(
                   model = "gpt-3.5-turbo",
                   messages = [{"role": "user", "content": "hi"}], 
                   max_tokens = 5
                   )
    except:
        return False
    else:
        return True
#%%
text_input_container = st.empty() # showcase bar
api_key = st.text_input('Insert OpenAi API key (this will be not registered anywhere):') # asking for key

printed = False # has the error message been printed?
while api_key != "" and printed == False: # if the user is inserting an input
    if IsKeyValid(api_key): # if the key is valid
        st.session_state['API_key'] = api_key   
        if IsKeyValid(st.session_state['API_key']):
            switch_page('ChatGPT') # open chat 
    else:
        printed = True # the error message has been printed
        st.info('The provided key is invalid') # ask for valid key
    


