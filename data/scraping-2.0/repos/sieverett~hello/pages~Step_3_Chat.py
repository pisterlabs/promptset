# Import required libraries
# from dotenv import load_dotenv
from itertools import zip_longest
import pandas as pd
import streamlit as st
from streamlit_chat import message
from utils import (capitalize_names, list_cloned_voices)
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from elevenlabs import Voice, generate, play, save, stream

  



# Load environment variables
# load_dotenv()

# Set streamlit page configuration
# st.set_page_config(page_title="ChatBot Starter")
# st.title("ChatBot Starter")

# Initialize the ChatOpenAI model
chat = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo",
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)
  

# Initialize session state variables
def initialize_session_state():
    # st.session_state['prompt_input'] = ""  # Store user input
    st.session_state['generated'] = []  # Store AI generated responses
    st.session_state['past'] = []  # Store past user inputs
    st.session_state['entered_prompt'] = ""  # Store the latest user input


# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []  # Store AI generated responses

# if 'past' not in st.session_state:
#     st.session_state['past'] = []  # Store past user inputs

# if 'entered_prompt' not in st.session_state:
#     st.session_state['entered_prompt'] = ""  # Store the latest user input


def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    df = pd.read_csv("TED_playlist_info.csv")
    transcript = df[df.presenter==selected_name_lower]['transcripts'].to_string(index=False)[400:]
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        content=f"You are {selected_name}. You are a TED Talk presenter who is answering questions about your presentation. If you do not know an answer, just say 'I don't know' or make a short joke, do not make up an answer.  Here's your presentation {transcript}")]

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg))  # Add user messages
        if ai_msg is not None:
            zipped_messages.append(
                AIMessage(content=ai_msg))  # Add AI messages

    return zipped_messages


def generate_response():
    """
    Generate AI response using the ChatOpenAI model.
    """
    # Build the list of messages
    zipped_messages = build_message_list()

    # Generate response using the chat model
    ai_response = chat(zipped_messages)

    audio_stream = generate("Hello world", stream=True)  


    v1=Voice.from_id(voice_id) # voice.voice_id
    audio_stream=generate(text=ai_response.content, stream=True, voice=v1, model="eleven_turbo_v2")
    for chunk in audio_stream:  
        stream(chunk)

    # play(audio)
    # st.audio(audio)

    return ai_response.content


# Define function to submit user input
def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input = ""


if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input

cloned_dict=list_cloned_voices()
cloned_names=list(cloned_dict.keys())
cloned_names_caps=capitalize_names(cloned_names)
cloned_names_caps=[n for n in cloned_names_caps if n not in ('Ray Dalio','Dalai Lama')]
selected_name = st.sidebar.selectbox("Select a TED Presenter", options=cloned_names_caps, on_change=initialize_session_state)
selected_name_lower = selected_name.lower()
voice_id = cloned_dict[selected_name_lower]


st.title(f"💬 Chat With {selected_name}")

# Create a text input for user
st.text_input('You: ', key='prompt_input', on_change=submit)

if selected_name not in st.session_state:
    # initialize_session_state()
    st.session_state['selected_name'] = selected_name

if st.session_state.entered_prompt != "":
    # Get user query
    user_query = st.session_state.entered_prompt

    # Append user query to past queries
    st.session_state.past.append(user_query)

    # Generate response
    output = generate_response()

    # Append AI response to generated responses
    st.session_state.generated.append(output)

# Display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display AI response
        message(st.session_state["generated"][i], key=str(i),logo=f'https://api.dicebear.com/7.x/initials/svg?seed={selected_name}')
        # Display user message
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user', logo='https://api.dicebear.com/7.x/shapes/svg?seed=You')
