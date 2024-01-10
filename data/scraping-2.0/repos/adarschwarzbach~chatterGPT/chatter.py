import azure.cognitiveservices.speech as speechsdk
import openai
import os
import threading
from queue import Queue
import time
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
from scipy.io.wavfile import write


speech_key, service_region = # YOUR AZURE SPEECH KEY AND REGION HERE
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

openai.api_key = # YOUR OPENAI API KEY HERE

def record_audio(output_filename, duration=3, fs=44100):
    """Record audio for a specified duration."""
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)  # Use channels=1 for mono recording
    sd.wait()  
    write(output_filename, fs, recording) 


def speech_to_text(audio, queue):
    audio_config = speechsdk.audio.AudioConfig(filename=audio)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_recognizer.recognize_once_async().get()
    transcription = result.text

    queue.put(transcription)
    print(transcription)

def text_to_speech(text, output_filename):
    # Convert text to speech using Azure
    audio_output = speechsdk.audio.AudioOutputConfig(filename=output_filename)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)

    result = speech_synthesizer.speak_text_async(text).get()


def chat_with_gpt(input_text, queue):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text},
        ]
    )
    
    print(response)
    queue.put(response['choices'][0]['message']['content'])


def play_audio(file_path):
    audio = AudioSegment.from_file(file_path, format="mp3")
    play(audio)


def main():
    # Create a queue for communication between threads
    queue = Queue()

    # First, record an audio clip
    record_audio('audio.wav', duration=5)

    # Then, transcribe the speech to text
    threading.Thread(target=speech_to_text, args=('audio.wav', queue)).start()

    # Get the transcription from the queue
    transcription = queue.get()

    # Send the transcription to GPT-3 and get the response
    threading.Thread(target=chat_with_gpt, args=(transcription, queue)).start()

    # Get the response from the queue
    response = queue.get()

    # Convert the response to speech
    threading.Thread(target=text_to_speech, args=(response, 'response.mp3')).start()

    # Wait for the TTS process to finish
    while threading.active_count() > 1:
        time.sleep(1)

    # Play the response
    play_audio('response.mp3')

if __name__ == '__main__':
    main()
