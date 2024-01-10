import os
from google.cloud import speech
import pvporcupine
import pyaudio
import wave
import struct
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException
import time
from pydub import AudioSegment
from pydub.playback import play
import random
import openai
from google.cloud import texttospeech
import pygame
import uuid

# Set Google Cloud credentials
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\nguye\\OneDrive - Cal Poly Pomona\\projects\\voiceassistantai\\voiceai.json"
openai.api_key = 'sk-redacted'

instructions = "You are an assistant that can perform tasks and answer questions. Since the response will be translated to speech, try to keep it short"

# Initialize Google Cloud Speech client
stt_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()
audio_responses = [
    "intro/readyforyourcommand.mp3",
    "intro/whatsupBoss.mp3",
    "intro/whattaskmayiperform.mp3",
    "intro/whatwouldyouhavemedo.mp3",
    "intro/yeshowmayihelpyou.mp3"
]


def main():
    detector = create_detector()
    print("Listening...")
    try:
        while True:
            if detect_wake_word(detector):
                play_audio(random.choice(audio_responses))
                audio_file = record_audio()
                transcribed_text = transcribe_audio(audio_file)
                print(f"Transcribed Text: {transcribed_text}")

                # Process transcribed text with GPT-4
                gpt_response = process_with_gpt(transcribed_text)
                print(f"GPT-3.5 Response: {gpt_response}")
                text_to_speech(gpt_response)


    finally:
        detector.delete()


def process_with_gpt(text):
    """
    Function to send text to GPT-3.5 Turbo and return the response.
    """
    try:
        # Combine instructions and user input to form the messages
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": text}
        ]

        # Call the API with the constructed messages
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=50  # Adjust the number of tokens (words) in the response
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error with GPT-3.5 Turbo processing: {e}")
        return ""


def play_audio(file_path):
    """Function to play an audio file."""
    sound = AudioSegment.from_file(file_path, format="mp3")
    play(sound)


user_pins = {}


def create_detector():
    access_key = 'redacted'
    keyword_file_path = "terminal.ppn"  # replace with the path to your .ppn file
    return pvporcupine.create(access_key=access_key, keyword_paths=[keyword_file_path])


def detect_wake_word(detector):
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=detector.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=detector.frame_length)
    try:
        while True:
            pcm = audio_stream.read(detector.frame_length)
            pcm = struct.unpack_from("h" * detector.frame_length, pcm)
            keyword_index = detector.process(pcm)
            if keyword_index >= 0:
                return True
    finally:
        audio_stream.close()
        pa.terminate()


def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    print("Recording...")
    frames = []
    for i in range(0, int(16000 / 1024 * 3)):  # Adjust recording duration as needed
        data = stream.read(1024)
        frames.append(data)
    print("Finished recording")

    # Save the recording as a WAV file
    file_path = "command.wav"
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

    stream.stop_stream()
    stream.close()
    p.terminate()

    return file_path


def transcribe_audio(file_path):
    with open(file_path, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US',
    )
    response = stt_client.recognize(config=config, audio=audio)

    # Check if response.results is not empty
    if response.results:
        # Return the transcribed text
        return response.results[0].alternatives[0].transcript
    else:
        # Handle the case where no transcription is returned
        print("No transcription available for the provided audio.")
        return ""


def text_to_speech(text):
    """
    Convert text to speech and play the audio.
    """
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name="en-US-Neural2-J", ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected voice parameters and audio file type
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    output_filename = f"output_{uuid.uuid4()}.mp3"

    # Write the response to the output file.
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to file "{output_filename}"')

    # Play the audio using pygame
    pygame.mixer.init()
    pygame.mixer.music.load(output_filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue


if __name__ == "__main__":
    main()