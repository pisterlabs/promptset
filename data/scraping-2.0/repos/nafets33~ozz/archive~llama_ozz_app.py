import streamlit as st
import pandas as pd
import numpy as np
from ozz.ozz_bee import send_ozz_call

import streamlit_chat
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
# from hugchat import hugchat
import openai
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.getcwd(), ".env"))

# st.set_page_config(page_title="ozz")

st.set_page_config(
    page_title="ozz",
    # page_icon=page_icon,
    # layout="wide",
    # initial_sidebar_state='collapsed',
    #  menu_items={
    #      'Get Help': 'https://www.extremelycoolapp.com/help',
    #      'Report a bug': "https://www.extremelycoolapp.com/bug",
    #      'About': "# This is a header. This is an *extremely* cool app!"
    #  }
)

st.title("Ozz, Your Learning Walk Guide")
def main():
    # Sidebar contents
    # with st.sidebar:
    #     st.title('🤗💬 HugChat App')
    #     st.markdown('''
    #     ## About
    #     This app is an LLM-powered chatbot built using:
    #     - [Streamlit](https://streamlit.io/)
    #     - [HugChat](https://github.com/Soulter/hugging-chat-api)
    #     - [OpenAssistant/oasst-sft-6-llama-30b-xor](https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor) LLM model
        
    #     💡 Note: No API key required!
    #     ''')
    #     add_vertical_space(5)
    #     st.write('Made with ❤️ by [Data Professor](https://youtube.com/dataprofessor)')

    # Generate empty lists for generated and past.
    ## generated stores AI generated responses
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hi! I am Ozz, ask me about Learning Walks!!"]
    ## past stores User's questions
    if 'past' not in st.session_state:
        st.session_state['past'] = ['']

    # Layout of input/response containers
    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    response_container = st.container()

    # User input
    ## Function for taking user provided prompt as input
    def get_text():
        input_text = st.text_input("You: ", "", key="input")
        return input_text
    ## Applying the user input box
    with input_container:
        user_input = get_text()

    # Response output
    ## Function for taking user prompt as input followed by producing AI generated responses
    # def generate_response(prompt):
    #     chatbot = hugchat.ChatBot()
    #     response = chatbot.chat(prompt)
    #     return response

    ## Conditional display of AI generated responses as a function of user provided prompts
    with response_container:
        if user_input:
            # response = generate_response(user_input)
            response = send_ozz_call(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)
            
        if st.session_state['generated']:
            for i in reversed(range(len(st.session_state['generated']))):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


# be sure to end each prompt string with a comma.
print('e', os.environ.get('ozz_api_key'))
# openai.api_key = os.environ.get('ozz_api_key')
openai.api_key = "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"

example_user_prompts = [
    "echo Hello World!",
    "How old is Elon Musk?",
    "What makes a good joke?",
    "Tell me a haiku.",
]


def move_focus():
    # inspect the html to determine which control to specify to receive focus (e.g. text or textarea).
    st.components.v1.html(
        f"""
            <script>
                var textarea = window.parent.document.querySelectorAll("textarea[type=textarea]");
                for (var i = 0; i < textarea.length; ++i) {{
                    textarea[i].focus();
                }}
            </script>
        """,
    )


def stick_it_good():

    # make header sticky.
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: white;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid black;
                }
            </style>
        """,
        unsafe_allow_html=True
    )


def userid_change():
    st.session_state.userid = st.session_state.userid_input
    
    
def complete_messages(nbegin,nend,stream=False, query=False):
    messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
    with st.spinner(f"Waiting for {nbegin}/{nend} responses from ChatGPT."):
        if stream:
            responses = [] # how to get responses?
            # Looping over openai's responses. async style.
            for response in openai.ChatCompletion.create(
                model = st.session_state["openai_model"],
                messages = messages,
                stream = True):
                    partial_response_content = response.choices[0].delta.get("content","")
                    responses.append(partial_response_content)
            response_content = "".join(responses)
        else:
            if query:
                print("ozzbee")
                response_content = send_ozz_call(query) # Send llama call
                if f"""I don't know.""" in response_content:
                    ozz_bee = "I'm not sure, what about..."
                    response = openai.ChatCompletion.create(model=st.session_state["openai_model"],
                                                            messages=[{"role": m["role"], "content": m["content"]}for m in st.session_state.messages],
                                                            stream=False
                                                            )
                    response_content = response.choices[0]['message'].get("content","")
                    response_content = ozz_bee + response_content
            else:
                response = openai.ChatCompletion.create(model=st.session_state["openai_model"],
                                                        messages=[{"role": m["role"], "content": m["content"]}for m in st.session_state.messages],
                                                        stream=False
                                                        )
                response_content = response.choices[0]['message'].get("content","")
    return response_content
    

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

stt_button = Button(label="Speak", width=100)

stt_button.js_on_event("button_click", CustomJS(code="""
    var recognition = new webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
 
    recognition.onresult = function (e) {
        var value = "";
        for (var i = e.resultIndex; i < e.results.length; ++i) {
            if (e.results[i].isFinal) {
                value += e.results[i][0].transcript;
            }
        }
        if ( value != "") {
            document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
        }
    }
    recognition.start();
    """))

result = streamlit_bokeh_events(
    stt_button,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)
print('r', result)
if result:
    if "GET_TEXT" in result:
        st.write(result.get("GET_TEXT"))

def main():


    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    with st.container():
        st.title("Streamlit ChatGPT Bot")
        stick_it_good()

    if "userid" in st.session_state:
        st.sidebar.text_input(
            "Current userid", on_change=userid_change, placeholder=st.session_state.userid, key='userid_input')
        if st.sidebar.button("Clear Conversation", key='clear_chat_button'):
            st.session_state.messages = []
            move_focus()
        if st.sidebar.button("Show Example Conversation", key='show_example_conversation'):
            #st.session_state.messages = [] # don't clear current conversaations?
            for i,up in enumerate(example_user_prompts):
                st.session_state.messages.append({"role": "user", "content": up})
                assistant_content = complete_messages(i,len(example_user_prompts))
                st.session_state.messages.append({"role": "assistant", "content": assistant_content})
            move_focus()
        for i,message in enumerate(st.session_state.messages):
            nkey = int(i/2)
            if message["role"] == "user":
                streamlit_chat.message(message["content"], is_user=True, key='chat_messages_user_'+str(nkey))
            else:
                streamlit_chat.message(message["content"], is_user=False, key='chat_messages_assistant_'+str(nkey))

        if user_content := st.chat_input("Type your question here."): # using streamlit's st.chat_input because it stays put at bottom, chat.openai.com style.
                nkey = int(len(st.session_state.messages)/2)
                streamlit_chat.message(user_content, is_user=True, key='chat_messages_user_'+str(nkey))
                st.session_state.messages.append({"role": "user", "content": user_content})
                assistant_content = complete_messages(0,1, query=user_content)
                streamlit_chat.message(assistant_content, key='chat_messages_assistant_'+str(nkey))
                st.session_state.messages.append({"role": "assistant", "content": assistant_content})
                #len(st.session_state.messages)
    else:
        st.sidebar.text_input(
            "Enter a random userid", on_change=userid_change, placeholder='userid', key='userid_input')
        streamlit_chat.message("Hi. I'm your friendly streamlit ChatGPT assistant.",key='intro_message_1')
        streamlit_chat.message("To get started, enter a random userid in the left sidebar.",key='intro_message_2')
                
if __name__ == '__main__':
    main()