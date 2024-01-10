import keyboard
import os
import tempfile
from dotenv import load_dotenv

import openai
import sounddevice as sd
import soundfile as sf

from elevenlabs import generate, play, set_api_key
from langchain.agents import initialize_agent, load_tools
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.utilities.zapier import ZapierNLAWrapper

load_dotenv()

set_api_key(os.environ['ELEVEN_LABS_API_KEY'])
openai.api_key = os.environ['OPENAI_API_KEY']

duration = 5
fs = 44100
channels = 1 

def record_audio(duration, fs, channels):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    print("Finished recording.")
    return recording

def transcribe_audio(recording, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, recording, fs)
        temp_audio.close()
        with open(temp_audio.name, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        os.remove(temp_audio.name)
    return transcript["text"].strip()

def play_generated_audio(text, voice="Bella", model="eleven_monolingual_v1"):
    audio = generate(text=text, voice=voice, model=model)
    play(audio)


if __name__ == '__main__':

    llm = OpenAI(temperature=0.6)
    memory = ConversationBufferMemory(memory_key="chat_history")

    zapier = ZapierNLAWrapper(zapier_nla_api_key=os.environ['ZAPIER_API_KEY'])
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

    tools = toolkit.get_tools() + load_tools(["human"])

    agent = initialize_agent(tools, llm, memory=memory, 
                             agent="conversational-react-description", verbose=True)

    while True:
        # print("Press spacebar to start recording.")
        # keyboard.wait("space")
        # recorded_audio = record_audio(duration, fs, channels)
        # message = transcribe_audio(recorded_audio, fs)
        message = input("You: ")
        assistant_message = agent.run(message)
        play_generated_audio(assistant_message)