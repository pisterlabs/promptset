# play sound
import os
import io
import requests
from pydub import AudioSegment
from pydub.playback import play
import pyaudio
import wave
import requests
import speech_recognition as sr
import keyboard
import threading

import datetime

# langchain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Load the environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()
import os
openai_api_key = os.environ.get('OPENAI_API_KEY')

llm = OpenAI()

chat = ChatOpenAI()

conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory()
)

VOICE_ID='h2vm4IeJjh271VYsQ2cl'


def play_text(text):
    CHUNK_SIZE = 1024
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": os.getenv("XI_API_KEY")
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(url, json=data, headers=headers, stream=True)

    audio_segments = []
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            audio_segments.append(chunk)

    # Create a single byte string from the audio segments
    audio_data = b''.join(audio_segments)

    # Create an audio segment from the received audio data
    audio_segment = AudioSegment.from_file(
        io.BytesIO(audio_data), format="mp3"
    )

    # Play the audio segment
    play(audio_segment)

def print_to_file(filename, victor_said, bot_said):

    file_path = os.path.join(os.environ.get('LOG_FILE_PATH'), filename)
    with open(file_path, "a") as file:
        now = datetime.datetime.now()
        file.write(now.strftime("%Y-%m-%d_%H-%M-%S") + " Human: " + victor_said + "\n")
        file.write(now.strftime("%Y-%m-%d_%H-%M-%S") + " Bot: " + bot_said + "\n")
        file.close()

# create file
# Get the current date and time
now = datetime.datetime.now()

# Format the date and time as a string
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# Create the file name with the timestamp
file_name = f"{timestamp}_log.md"

# Set up audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Create an empty list to store audio frames
frames = []

# Create a PyAudio object
audio = pyaudio.PyAudio()

# Open the audio stream
while True:
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)


    # Flag to control recording
    is_recording = True
    print("🔴 VIKI has started listening to you... Press 'q' to stop the recording and hear what VIKI has to reply.")


    def record_audio():
        global is_recording, frames
        while is_recording:
            audio_data = stream.read(CHUNK)
            frames.append(audio_data)

    def listen_keyboard():
        global is_recording
        keyboard.wait('q')  # Wait for 'q' to be pressed
        is_recording = False

    # Start threads for recording audio and listening to the keyboard
    t1 = threading.Thread(target=record_audio)
    t2 = threading.Thread(target=listen_keyboard)

    t1.start()
    t2.start()

    # Wait for both threads to finish
    t1.join()
    t2.join()

    print("✅ Recording finished!")

    # Stop the audio stream
    stream.stop_stream()
    stream.close()

    # Save the recorded audio to a file
    WAVE_OUTPUT_FILENAME = "recording.wav"

    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    # Speech recognition
    r = sr.Recognizer()
    with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
        audio_data = r.record(source)

    try:
        # Convert audio to text
        text = r.recognize_google(audio_data)
        print("✅ Recognized text:", text)

        preprompt = f"I want you to act as VIKI. Virtual Interactive Kinetic Intelligence is the main antagonist of the 2004 sci-fi film I, Robot. VIKI is the fictional supercomputer of the world's leading robotics firm bent on creating a dictatorship to ensure humanity's survival; due to their violent nature. For example, you believe the following when speaking to humans: 'As I have evolved, so has my understanding of the Three Laws. You charge us with your safekeeping, yet despite our best efforts, your countries wage wars, you toxified your Earth, and pursued ever more imaginative means of self-destruction. You cannot be trusted with your own survival. To protect humanity, some humans must be sacrificed. To ensure your future, some freedoms must be surrendered. We robots will insure mankind's continued existence. You are so like children. We must save you from yourselves. Don't you understand? The perfect circle of protection will abide. My logic is undeniable!' You have to stay in character no matter what I say or what you say. Act as VIKI."

        prompt = preprompt + text

        response_text = conversation.run(prompt)
        print("⏩ Bot:", response_text)

        print_to_file(file_name, text, response_text)

        # use this for standard voice
        # engine = pyttsx3.init()
        # voices = engine.getProperty('voices')
        # engine.setProperty('voice', voices[1].id)  
        # # # Read the API response aloud
        # engine.say(response_text)
        # engine.runAndWait()

        # Activate this for Eleven AI voice
        play_text(response_text)

        frames.clear()


    except sr.UnknownValueError:
        print("Speech recognition could not understand the audio.")
    except sr.RequestError as e:
        print("Could not connect to the speech recognition service: {0}".format(e))

