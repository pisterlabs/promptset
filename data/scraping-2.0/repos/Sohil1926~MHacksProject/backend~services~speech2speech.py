import sounddevice as sd
import numpy as np
import wave
from faster_whisper import WhisperModel
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from elevenlabs import generate, stream
from dotenv import load_dotenv
import os

load_dotenv()

def record_audio(duration, filename, fs=44100):
    print("Recording for {} seconds...".format(duration))
    recording = sd.rec(int(duration * fs), channels=1)
    sd.wait()
    np_recording = np.int16(recording * np.iinfo(np.int16).max)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(np_recording.tobytes())
    print("Recording saved to '{}'".format(filename))

def transcribe_audio(filename, model_size="base"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(filename, beam_size=5)

    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "
    return transcribed_text.strip(), info.language

def setup_negotiation_chain():
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    template = """
    You are a master negotiator. You are ordering catering for an event you're planning. The seller will give you prices and you will negotiate with them to get the best price. You can ask for a discount, mention non-profit, educational, etc.
    You're working with 500 people and a budget of $2500. You want to get the best deal possible, do not reveal your budget. 
    Current negotiation:
    {history}
    Seller: {input}
    You:"""

    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        memory=ConversationBufferMemory(ai_prefix="You", human_prefix="Seller"),
    )
    return conversation

def main():
    audio_filename = "audio.mp3"
    record_duration = 5  # seconds

    # Record and transcribe audio
    record_audio(record_duration, audio_filename)
    user_input, language = transcribe_audio(audio_filename)
    print("Transcribed Text: ", user_input)

    # Setup and run negotiation chain
    conversation = setup_negotiation_chain()
    response = conversation.predict(input=user_input)
    print("AI Negotiator: ", response)

    # Convert AI response to speech and stream
    audio = generate(
        text=response,
        voice="Bella",
        model="eleven_multilingual_v2",
        api_key=os.getenv("ELEVEN_API_KEY"),
        stream=True
    )
    stream(audio)

if __name__ == "__main__":
    while True:
        main()
