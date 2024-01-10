#   Allows you to have (lagged) but live time 
#   conversations with a bot who loves trains

from time import sleep
import sounddevice as sd
import numpy
from scipy.io.wavfile import write
import datetime
import openai 
import os
import requests
from playsound import playsound
############## Constants ###############
chatbot_voice_id1 = "AZnzlk1XvdvUeBnXmlld"
url = f"https://api.elevenlabs.io/v1/text-to-speech/{chatbot_voice_id1}"



############### Add API keys  ###############

openai.api_key=os.environ["OPENAI_KEY"]
eleven_labs_api_key = os.environ["ELEVEN_LABS_API_KEY"]

############### Set up your chatbot's personality ###############
messages=[
    {"role": "system", "content": "Your name is JoeShmoe. You are a friend to the person talking. You also like trains, and feel compelled to let everyone know. Please act as if you are a friend to the person talking. You are not an AI model, you are a human being with feelings (and a train obsession). Please keep your answers rather short, if possible.)"},
    ]

############### Tuning variables ###############

max_conversation_iterations = 10
max_conversation_time_seconds = 15
recoring_period = 1.25
audio_input_sample_rate = 44100  
quiet_threshold = 150 # If the sound level is below this threshold, the recording will stop
                      # use sum(abs(myrecording)) to get the sound level, and test the threshold for your
                      # environment
current_loop_iteration = 0

# Handle running iterations of the conversation
# Quit using ctrl-c 
def main():
    loop = 0
    while ( loop < max_conversation_iterations):
        audio_file = record_audio()
        transcription = transcribe_audio(audio_file)
        print("Transcription: ", transcription)
        messages.append({"role": "user", "content": transcription})
        chatbot_response = get_chatbot_response()
        print("Response: ", chatbot_response)
        messages.append({"role": "assistant", "content": chatbot_response})
        have_chatbot_speak(chatbot_response)
        loop += 1
    
# Record the microphone's sound.
def record_audio():
    long_recording = []
    sound_level = 1000
    time_recorded = 0 
    print("Begin recording in 3...")
    sleep(0.5)
    print("2...")
    sleep(0.5)
    print("1...")
    sleep(0.5)
    while (sound_level > quiet_threshold and time_recorded < max_conversation_time_seconds):
        print("Listening . . .")
        # Record the audio
        myrecording = sd.rec(int(recoring_period * audio_input_sample_rate), 
                             samplerate=audio_input_sample_rate, 
                             channels=1)
        
        sd.wait() # Wait for the conversation to end
        
        # Check if they stopped talking
        sound_level = sum(abs(myrecording)) / recoring_period

        # Add the samples to one continuous sample
        long_recording.extend(myrecording)
        time_recorded += 1

        # write to the output file
    long_recording = numpy.array(long_recording)
    write('.TEMP-output.wav', audio_input_sample_rate, long_recording)  # Save as WAV file 
    audio_file = open(".TEMP-output.wav", "rb")
    return audio_file # test


# Transcribe the audo file into text using OpenAI Whisper API
def transcribe_audio(audio_file):
    print("Transcribing audio . . .")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]
    
# Get chatbot response from OpenAI GPT-3.5-turbo API
def get_chatbot_response():
    print("Asking openAI gpt-3.5-turbo for a response . . .")
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    )
    
    return response["choices"][0]["message"]["content"]

# Takes the text and has the chatbot speak it through the computer speakers
# uses eleven labs API 
def have_chatbot_speak(text):
    print("Requesting chatbot voice . . .")
    request_body = {
        "text": text,
        "voice_settings": {
            "stability": 0.8,
            "similarity_boost": 0.1,
        }
        }
    headers = {
        "xi-api-key": eleven_labs_api_key,
        "Content-Type": "application/json"
    }
    
    
    # print("Asking Eleven Labs for an audio file . . .")
    audio = requests.post(url, headers=headers, json=request_body) 
    with open("audio.mp3", "wb") as f:
        f.write(audio.content)
    print("Playing audio . . .")
    playsound("audio.mp3")

main()