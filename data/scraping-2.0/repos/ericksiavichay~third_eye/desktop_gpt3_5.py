import os
import cv2
import requests
import openai
import io
import asyncio
import threading
import wave
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
from elevenlabs import set_api_key
from deepgram import Deepgram
import aiohttp

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = "1da2bb9407e638f4445d437c8e8770e1"


def play_audio(audio_bytes):
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        play(audio)
    except Exception as e:
        print(f"Failed to play audio: {e}")
        
# Function to transcribe audio via Deepgram
async def transcribe_audio(audio_bytes, DEEPGRAM_API_KEY='5a1672a5d83a9b723a45375eb0077e5c464a8909'):
    deepgram = Deepgram(DEEPGRAM_API_KEY)
    
    source = {
        'buffer': audio_bytes,
        'mimetype': 'audio/mp3'
    }

    async with aiohttp.ClientSession() as session:
        response = await asyncio.create_task(
            deepgram.transcription.prerecorded(
                source,
                {
                    'smart_format': True,
                    'model': 'nova',
                },
                session=session  # passing the session here
            )
        )
        
    return response["results"]["channels"][0]["alternatives"][0]["transcript"]

openai.api_key = OPENAI_API_KEY

# Function to get response from OpenAI API
def get_openai_response(prompt):
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt = (
            "Transform the provided image description: {" + prompt + "} into a "
            "succinct and actionable auditory guide for visually impaired users utilizing "
            "smart glasses. Ensure the information is precise, factual, and contextually relevant, "
            "highlighting immediate obstacles, noteworthy objects, and essential environmental "
            "details. Prioritize user safety by identifying any potential hazards and suggesting "
            "immediate actions when necessary. The information should be straightforward, "
            "easily comprehensible, and directly applicable through auditory means."
        ),
            max_tokens=500,
            temperature=0.7
        )
        message = response['choices'][0]['text'].strip()
        return message
    except Exception as e:
        return str(e)

set_api_key("1da2bb9407e638f4445d437c8e8770e1")

# Function to get text to speech from ElevenLabs
def get_tts(text):
    tts_url = "https://api.elevenlabs.io/v1/text-to-speech/D38z5RcWu1voky8WS1ja"
    tts_headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_API_KEY
    }
    tts_data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    try:
        tts_response = requests.post(tts_url, json=tts_data, headers=tts_headers)
        tts_response.raise_for_status()
        return tts_response.content
    except requests.RequestException as e:
        print(f"Failed to communicate with ElevenLabs API: {e}")
        return None

# Global variables
is_recording = False  
audio_frames = []

# Function to record audio until a global flag is set
def record_audio():
    global is_recording
    global audio_frames

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    is_recording = True
    audio_frames = []

    while is_recording:
        data = stream.read(1024)
        audio_frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Press space to capture, and Esc to exit', frame)
    key = cv2.waitKey(1)

    if key == 27:  # Esc key
        break
    elif key == 32:  # Spacebar
        _, buffer = cv2.imencode('.jpg', frame)
        byte_img = buffer.tobytes()
        image_files = {'file': ('image.jpg', byte_img, 'image/jpeg')}
        
        try:
            # Send image to the relevant API
            image_response = requests.post('https://3942-64-247-206-81.ngrok-free.app/uploadfile', files=image_files)
            image_response.raise_for_status()
            response_json = image_response.json()
            description = response_json.get('text', "No description provided by the API")
            print(f"Description from API: {description}")
            # Get response from OpenAI API
            openai_response = get_openai_response(description)
            print(f"OpenAI GPT response: {openai_response}")

            # Convert description to speech via ElevenLabs API
            audio_bytes = get_tts(openai_response)
            if audio_bytes:
                play_audio(audio_bytes)
                
                # Transcribe audio via Deepgram
                transcription = asyncio.run(transcribe_audio(audio_bytes))
                if transcription:
                    print(f"============")
                    print(f"Environment description: {transcription}")
                    print(f"============")
                    print(f">>>")


            # Start recording thread
            record_thread = threading.Thread(target=record_audio)
            record_thread.start()

            # Instruction for the user
            print("Press the spacebar again to stop recording...")

            # Wait for user to press a key to stop recording
            while True:
                if cv2.waitKey(1) == 32:  # Spacebar
                    is_recording = False
                    record_thread.join()  # Ensure the recording thread finishes
                    break
                
            # Save the recorded audio to a .wav file
            wf = wave.open('recorded_audio.wav', 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(audio_frames))
            wf.close()

            # Assume the transcription API accepts a file. Adjust if needed.
            with open('recorded_audio.wav', 'rb') as audio_file:
                transcription = asyncio.run(transcribe_audio(audio_file.read()))
                if transcription:
                    print(f"============")
                    print(f"Transcription of user: {transcription}")
                    print(f"============")
                    
                    # Constructing new prompt for OpenAI API
                    new_prompt = (
                        "Provided image description: {" + description +
                        "} and the user question {" + transcription +
                        "} provide a succinct answer "
                    )
                    
                    # Get response from OpenAI API using the new prompt
                    openai_response = get_openai_response(new_prompt)
                    print(f"OpenAI GPT response using transcription: {openai_response}")
                    
                    # Convert openai_response to speech via ElevenLabs API
                    response_audio_bytes = get_tts(openai_response)
                    if response_audio_bytes:
                        play_audio(response_audio_bytes)

        except requests.RequestException as e:
            print("Failed to communicate with the description API.")
            print(e)

cap.release()
cv2.destroyAllWindows()
