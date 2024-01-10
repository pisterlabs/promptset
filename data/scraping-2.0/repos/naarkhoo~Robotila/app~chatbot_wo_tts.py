"""Here is the main chatbot."""
import base64
import os
from io import BytesIO

import streamlit as st
from google.cloud import texttospeech
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from utils import prepare_tts, response_to_audio

# Set the environment variable in your code (not recommended for production)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds/text-to-speech-key.json"

# Instantiates a client
client_tts = texttospeech.TextToSpeechClient()
voice, audio_config = prepare_tts()

st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory")

# Set an OpenAI API Key before continuing
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Set up the LLMChain, passing in memory
template = """You are an AI chatbot having a conversation with a human.

{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
llm_chain = LLMChain(
    llm=OpenAI(openai_api_key=openai_api_key), prompt=prompt, memory=memory
)


# Function to create an audio player with autoplay enabled
def create_autoplay_audio_player(
    audio_file: BytesIO, file_type: str = "audio/mp3"
) -> str:
    """Create an audio player with autoplay enabled."""
    base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")
    audio_html = f"""
        <audio autoplay>
            <source src="data:{file_type};base64,{base64_audio}" type="{file_type}">
            Your browser does not support the audio element.
        </audio>
    """
    return audio_html


# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    response = llm_chain.run(prompt)
    st.chat_message("ai").write(response)

    # Synthesize speech for the AI's response
    audio_file = response_to_audio(response, voice, audio_config, client_tts)

    # Create the audio player HTML with autoplay
    audio_html = create_autoplay_audio_player(audio_file)

    # Use the Streamlit `st.markdown` function to render the audio player in the app
    st.markdown(audio_html, unsafe_allow_html=True)

# View the message contents in session state
view_messages = st.expander("View the message contents in session state")
with view_messages:
    view_messages.json(st.session_state.langchain_messages)
