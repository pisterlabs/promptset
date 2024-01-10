import os
from flask import Flask, render_template, request, jsonify, send_file
import openai
import config
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from google.cloud import texttospeech
import base64
from langdetect import detect
from pvporcupine import Porcupine
import struct
import threading
import queue
import time
import GmailAPI
import re
###########################################################################
# Initialize Google Cloud Text-to-Speech client
tts_client = texttospeech.TextToSpeechClient()
openai.api_key = config.OPENAI_API_KEY
porcupine_accesskey = config.PORCUPINE_ACCESSKEY
wake_word_queue = queue.Queue()
###########################################################################
messages = []
app = Flask(__name__, static_folder="static", template_folder="templates")

###########################################################################
"""Transcription part of the code"""
# Function that transcribes audio using OpenAI's API


def transcribeAudio(audio):
    global model_id
    media_file_path = audio
    media_file = open(media_file_path, "rb")

    response = openai.Audio.transcribe(
        model="whisper-1",
        file=media_file,
        response_format="json"
    )

    prompt = response["text"]
    return prompt


###########################################################################


###########################################################################
def gptPrompt(prompt):
    global messages

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Your name is Jarvis. You're an AI assistant that helps with everyday tasks. You can do things like send emails, play music, send messages, and more.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=4096,
    )
    gptResponse = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": gptResponse})

    return gptResponse


###########################################################################
duration = 10  # Record for x seconds
fs = 44100  # Sample rate (44.1 kHz)
channels = 1  # Use mono
output_filename = "voice_recording.wav"

# Function to create wav file


def write_wav(data, filename, fs):
    with open(filename, 'wb') as file:
        file.write(b'RIFF')
        file.write((36 + len(data)).to_bytes(4, 'little'))
        file.write(b'WAVEfmt ')
        file.write((16).to_bytes(4, 'little'))
        file.write((1).to_bytes(2, 'little'))
        file.write((channels).to_bytes(2, 'little'))
        file.write((fs).to_bytes(4, 'little'))
        file.write((fs * channels * 2).to_bytes(4, 'little'))
        file.write((4).to_bytes(2, 'little'))
        file.write((16).to_bytes(2, 'little'))
        file.write(b'data')
        file.write((len(data)).to_bytes(4, 'little'))
        file.write(data)


def voiceRecorder():
    # Recording
    print("Recording started...")
    recorded_data = sd.rec(
        int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()  # Wait for the recording to finish
    print("Recording finished.")

    # Convert data to PCM format and save as WAV file
    pcm_data = (recorded_data *
                np.iinfo(np.int16).max).astype(np.int16).tobytes()
    write_wav(pcm_data, output_filename, fs)
    print(f"Voice recording saved as {output_filename}.")
    # transcribe audio file
    fileText = transcribeAudio(output_filename)
    return fileText
###########################################################################
# Function to convert text to speech


def text_to_speech(text, language):
    # Initialize a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)
    languageCode = get_language_code(text)
    # Set the voice parameters
    voice = texttospeech.VoiceSelectionParams(
        language_code=language, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL, name=f"{languageCode}-Standard-A")

    # Set the type of audio file you want to return
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    # Create the speech synthesis request
    response = client.synthesize_speech(input=synthesis_input,
                                        voice=voice,
                                        audio_config=audio_config)

    return response.audio_content


###########################################################################
language_code_cache = {}  # cahce variable for the language code to reuse


def get_language_code(text):
    # Detect language from text
    language = detect(text)

    # If the result is in the cache, return it
    if language in language_code_cache:
        return language_code_cache[language]

    # If not then proceed to find the language
    # Map language to appropriate language code for Google Text to Speech
    lang_code_map = {
        'en': 'en-US',  # English
        'ja': 'ja-JP',  # Japanese
        'fr': 'fr-FR',  # French
        'es': 'es-ES',  # Spanish
        'de': 'de-DE',  # German
        'it': 'it-IT',  # Italian
        'ko': 'ko-KR',  # Korean
        # Add other languages and their corresponding Google Text to Speech codes here
    }

    # If language is not supported, default to English
    language_code = lang_code_map.get(language, 'en-US')

    return language_code


###########################################################################

# Introduce a flag at the beginning of your script
processing_wake_word = False

# Paths for Porcupine's library, model and keyword files.
# Adjust to the actual path
LIBRARY_FILE_PATH = '/Users/hosseinhajmirbaba/Desktop/Txt_Speech/myenv/lib/python3.9/site-packages/pvporcupine/lib/mac/arm64/libpv_porcupine.dylib'
# Adjust to the actual path
MODEL_FILE_PATH = '/Users/hosseinhajmirbaba/Desktop/Txt_Speech/myenv/lib/python3.9/site-packages/pvporcupine/lib/common/porcupine_params.pv'
# Adjust to the actual path(s)
KEYWORD_FILE_PATHS = [
    '/Users/hosseinhajmirbaba/Desktop/Txt_Speech/myenv/lib/python3.9/site-packages/pvporcupine/resources/keyword_files/mac/jarvis_mac.ppn']

# List of keywords to listen for.
keywords = ["jarvis"]


running = True

conversation_in_progress = False


def porcupine_listener():
    global running
    global conversation_in_progress
    import pyaudio
    porcupine = None
    audio_stream = None

    try:
        porcupine = Porcupine(
            access_key=porcupine_accesskey,
            library_path=LIBRARY_FILE_PATH,
            model_path=MODEL_FILE_PATH,
            keyword_paths=KEYWORD_FILE_PATHS,
            sensitivities=[0.5] * len(KEYWORD_FILE_PATHS))

        audio_stream = pyaudio.PyAudio().open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length)

        print("AI is actively listening...")

        while running:
            try:
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                # print(f"conversation_in_progress: {conversation_in_progress}")

                if not conversation_in_progress:
                    keyword_index = porcupine.process(pcm)

                    if keyword_index >= 0:
                        print("Wake word detected again!")
                        wake_word_detected(KEYWORD_FILE_PATHS[keyword_index])
                        time.sleep(1)
            except IOError as e:
                print(f"IOError caught: {e}")
                continue  # continue to the next iteration of the loop
            except OSError as e:
                print(f"OSError caught: {e}")
                if e.errno == -9981:
                    # input overflowed, possibly skip a short time to recover
                    time.sleep(0.1)
                    continue  # continue to the next iteration of the loop

    except Exception as e:
        print(e)
        raise e

    finally:
        if porcupine is not None:
            porcupine.delete()

        if audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()

        pyaudio.PyAudio().terminate()


def play_greeting():
    greeting_text = "Yes Mr. Stark?"
    greeting_audio = text_to_speech(greeting_text, 'en')
    play_audio(greeting_audio)
    print(greeting_text)


def play_audio(audio_data):
    import sounddevice as sd
    import numpy as np
    import io
    import wave

    # Convert byte data to numpy array
    byte_stream = io.BytesIO(audio_data)
    with wave.open(byte_stream, 'rb') as wave_file:
        signal = np.frombuffer(wave_file.readframes(
            wave_file.getnframes()), dtype=np.int16)
        fs = wave_file.getframerate()

    # Play the audio using sounddevice
    sd.play(signal, samplerate=fs)
    sd.wait()


conversation = ""

# 18006677972


def wake_word_detected(keyword):
    global conversation
    global conversation_in_progress
    print('Jarvis detected!')
    play_greeting()
    # Set the flag to True when the conversation starts
    conversation_in_progress = True
    while True:
        # Record user's voice
        user_voice_data = voiceRecorder()
        # Convert user's voice to text
        # What the user said
        print(f"User said: {user_voice_data}")
        # If the user says "Stop" or "Thank you", break the loop
#      if "Thank you" in user_voice_data or "Stop" in user_voice_data:
      #      print("Stopping command detected. Exiting loop.")
       #     # Set the flag to False when the conversation ends
        #    conversation_in_progress = False
        #   break''
        conversation += f"User: {user_voice_data}\n"
        # Get response from ChatGPT using gptPrompt function
        chatbotResponse = gptPrompt(conversation)
        conversation += f"User: {chatbotResponse}\n"
        chatbot_lang_code = get_language_code(chatbotResponse)
        # Convert the chatbot response to audio and play it
        response_audio = text_to_speech(chatbotResponse, chatbot_lang_code)
        # What the chatbot said
        print(f"Chatbot said: {response_audio}")
        play_audio(response_audio)


###########################################################################


def extract_email_details(user_input):
    # Detect if it's an email command
    if "write an email to" not in user_input:
        return None

    # Extract email address
    email_pattern = re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}")
    email_match = email_pattern.search(user_input)
    if not email_match:
        return None
    email_address = email_match.group()

    # Extract subject
    subject_match = re.search(r"with the subject of ([^and]+)", user_input)
    if not subject_match:
        return None
    subject = subject_match.group(1).strip()

    # Extract content message to be passed to GPT-4
    content_match = re.search(r"tell that person that (.+)$", user_input)
    if not content_match:
        return None
    content_prompt = content_match.group(1).strip()

    # Pass the content_prompt to GPT-4 to generate the full content
    # For this example, we'll assume a function gpt4_generate is available
    content = gptPrompt(content_prompt)

    return {
        "recipient": email_address,
        "subject": subject,
        "content": content  # or use "content": content if using GPT-4
    }

###########################################################################


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/record", methods=["POST"])
def record_voice():
    audio_data = request.files["audio_data"]
    audio_data.save("voice_recording.webm")

    # Get the language from the form data
    language = request.form.get('language', 'en')

    # Convert webm to wav
    webm_audio = AudioSegment.from_file("voice_recording.webm", format="webm")
    webm_audio.export("voice_recording.wav", format="wav")

    prompt = transcribeAudio("voice_recording.wav")

    response_text = gptPrompt(prompt)

    language_code = get_language_code(response_text)

    response_audio = text_to_speech(response_text, language_code)
    response_audio_b64 = base64.b64encode(response_audio).decode('utf-8')

    return jsonify({"response": response_text,
                    "userPrompt": prompt,
                    "response_audio": response_audio_b64})


@app.route('/check-wake-word', methods=['GET'])
def check_wake_word():
    if not wake_word_queue.empty():
        wake_word_queue.get()
        return jsonify({"wakeWordDetected": True})
    return jsonify({"wakeWordDetected": False})


###########################################################################
if __name__ == '__main__':
    threading.Thread(target=porcupine_listener).start()
    app.run(debug=True, use_reloader=False)
