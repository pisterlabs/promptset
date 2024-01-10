#### Please keep this in order at the top ####
# If you are running it locally, you can comment this out # 
# This is for the streamlit deployment #
#
# Use save without formatting if it's being automatically changed by formatting tool #
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#
##############################################

import streamlit as st
from langchain.memory import ConversationBufferMemory
from dataprocess import return_answer, load_default_resources, default_resources, generate_email_format_answer, check_response_before_answer, translate_to_selected_response_language
from dotenv import load_dotenv, find_dotenv
import openai
import os
from audio_recorder_streamlit import audio_recorder
from tempfile import NamedTemporaryFile
import whisper
import time



_ = load_dotenv(find_dotenv())  # read local .env file


st.set_page_config(page_title="SFBU BOT",
                   page_icon="./images/logo_bot.svg",
                   initial_sidebar_state="auto",
                   )

openai.api_key = os.environ['OPENAI_API_KEY']
openai_models = ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview']

client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=openai.api_key,
)
default_max_tokens = 500

###### Header UI Start ###################################
left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image('./images/jolly.png')
    st.header("SFBU ChatBot", anchor=False, divider="rainbow")
##########################################################


###### Define Default State Variable #####################

@st.cache_resource
def load_resources():
    return load_default_resources(load_from_local_stored_files=True)
# Load default resources
if 'default_vectorstore' not in st.session_state:
    st.session_state.default_vectorstore = load_resources()

# Initialise default retriver variable
if 'retriever' not in st.session_state:
    st.session_state.retriever = st.session_state.default_vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5})

# Initialise memory variable
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", max_len=20, return_messages=True)

# Initialise question_submit_clicked variable
if 'question_submit_clicked' not in st.session_state:
    st.session_state.question_submit_clicked = False

# Initialise answer_in_email_format_clicked variable
if 'answer_in_email_format_clicked' not in st.session_state:
    st.session_state.answer_in_email_format_clicked = False

# Initialise chat history variable
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialise model variable
if 'model' not in st.session_state:
    st.session_state.model = openai_models[0]

# Initialise temperature variable
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.0

if 'audio_recorder_key' not in st.session_state:
    st.session_state.audio_recorder_key = "1"

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = load_whisper_model()


# Initialise default response language variable
# The first one is Default, which means the default response language is same as the input language
response_languages = ["Default", "English",
                      "Spanish", "French", "Chinese", "Arabic"]
if 'response_language' not in st.session_state:
    st.session_state.response_language = response_languages[0]

##########################################################


##### Sidebar UI Start ###################################

with st.sidebar:
    cl1, cl2, cl3 = st.columns([1, 2, 1])
    with cl2:
        st.image('./images/logo_bot.svg')

    st.markdown("# SFBU BOT CONFIGURATION")
    st.divider()
    st.write('Choose OpenAI Model')
    # default to the first model
    # Add a label to avoid the warning and use label_visibility="collapsed" to hide the label
    model = st.selectbox('Select your model',
                         openai_models, index=openai_models.index(st.session_state.model), label_visibility="collapsed")
    with st.container():
        st.markdown(
            """<div>
                <div><small>GPT-3.5 : Less powerful, Faster response</small></div>
                <div><small>GPT-4.0 : More powerful, Slower response</small></div>
                </div>
            """, unsafe_allow_html=True)

    # set selected value back to session state
    st.session_state.model = model
    # print(f"st.session_state.model: {st.session_state.model}")

    st.divider()
    # Check session state first
    st.write('Choose Temperature')
    # default to the default temprature
    # Add a label to avoid the warning and use label_visibility="collapsed" to hide the label
    temperature = st.slider('Select your Temperature', min_value=0.0,
                            max_value=2.0, step=0.01, value=st.session_state.temperature, label_visibility="collapsed")

    with st.container():
        st.markdown(
            """<div>
                <div><small>Lower temperature : more deterministic results, higher accuracy</small></div>
                <div><small>Higher temperature : more creative results, lower accuracy</small></div>
                </div>
            """, unsafe_allow_html=True)

    # # set selected value back to session state
    # st.session_state.temperature = temperature
    # # print(f"st.session_state.temperature: {st.session_state.temperature}")

    st.divider()
    st.write('Choose Response Language')
    response_language = st.selectbox('Select your response language',
                                     response_languages, index=0, label_visibility="collapsed")
    with st.container():
        st.markdown(
            """<div>
                <div><small>By default, the response language is the same as the input language.</small></div>
                </div>
            """, unsafe_allow_html=True)

    # set selected value back to session state
    st.session_state.response_language = response_language
    # print(f"st.session_state.response_language: {st.session_state.response_language}")

    st.divider()

##########################################################


def generate_answer():
    st.session_state.question_submit_clicked = True


def generate_answer_in_email():
    st.session_state.answer_in_email_format_clicked = True


def result_all_button_state():
    st.session_state.question_submit_clicked = False
    st.session_state.answer_in_email_format_clicked = False
    st.session_state.query = ''


def clear_chat_history():
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", max_len=20, return_messages=True)


def generate_audio(input):

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=input
    )

    response.stream_to_file("output.mp3")

    st.audio("output.mp3")


##### Main UI and Logic ###################################

##### Voice to Text #######

if 'text_received' not in st.session_state:
    st.session_state.text_received = ""

audio_bytes = audio_recorder(text="", pause_threshold=1.5, key="audio",
                             sample_rate=60000, energy_threshold=0.003, icon_size="2x")
text = ""
if audio_bytes is not None:
    # Save the audio bytes to a temporary file
    with NamedTemporaryFile(delete=False, suffix='.wav') as f:
        f.write(audio_bytes)
        temp_audio_path = f.name
    # Load the Whisper model and transcribe the audio file
    model_audio = st.session_state.whisper_model
    result_text = model_audio.transcribe(temp_audio_path)
    text = result_text["text"]
if text:
    st.session_state.text_received = text

##########################

##### Textbox & Question Submit Button #######

st.session_state.query = st.text_input(label="Ask a question...",
                                       type="default", autocomplete="off", value=st.session_state.text_received)

c1, c2, c3 = st.columns([3, 3, 2])
with c1:
    st.button(label="Generate an answer", type="secondary",
              disabled=False, use_container_width=True, on_click=generate_answer)
with c2:
    st.button(label="Generate answer in email format", type="secondary",
              disabled=False, use_container_width=True, on_click=generate_answer_in_email)
with c3:
    st.button(label="Clear chat history", type="secondary",
              disabled=False, use_container_width=True, on_click=clear_chat_history)

#############################################

# After submitting the question
if st.session_state.question_submit_clicked or st.session_state.answer_in_email_format_clicked:

    # reset qa variable
    st.session_state.qa = return_answer(
        st.session_state.temperature, st.session_state.model, st.session_state.memory, st.session_state.retriever)

    query = st.session_state.query

    print("User submitted question --> : ", query)

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Generating Answer..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # Call the QA function with the necessary parameters to retrieve the initial resposne
        first_result = st.session_state.qa({"question": query})['answer']

        print("First result --> : ", first_result)

        if st.session_state.question_submit_clicked:

            final_response = check_response_before_answer(
                client, query, first_result, model, temperature, default_max_tokens)

            print("check_response_before_answer, final result --> : ", final_response)

        elif st.session_state.answer_in_email_format_clicked:
            temp_messages = st.session_state.messages.copy()
            temp_messages.append({"role": "assistant", "content": first_result})
            final_response = generate_email_format_answer(
                client, temp_messages, model, temperature)

            print("generate_email_format_answer, final result --> : ", final_response)

        if st.session_state.response_language != "Default":
            final_response = translate_to_selected_response_language(client,
                                                                     final_response, st.session_state.response_language, model, temperature)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = final_response
        full_response = '<div>' + full_response + '</div>'
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.02)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(
                full_response + "|", unsafe_allow_html=True)
        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.divider()
        if st.session_state.question_submit_clicked:
            with st.spinner("Generating Audio..."):
                generate_audio(final_response)
        else:
            pass

    # Add assistant message to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": final_response})

    # Reset the button state to wait for the next question
    result_all_button_state()
    # st.session_state.text_received = ""


##########################################################

##### Dispaly Previous Chat History #######
st.divider()

with st.expander("Click to view chat history"):
    # Display chat messages from history on app rerun
    # skip the current question and answer pair which is the last one
    for message_idx in range(len(st.session_state.messages)-3, -1, -1):
        message = st.session_state.messages[message_idx]
        with st.chat_message(message["role"]):
            st.markdown('<div>' + message["content"] +
                        '</div>', unsafe_allow_html=True)
    ##########################
