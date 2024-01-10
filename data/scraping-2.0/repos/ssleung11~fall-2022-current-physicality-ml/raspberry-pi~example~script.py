import RPi.GPIO as GPIO
import speech_recognition as sr
import os
import openai 
import time 
import pyaudio
import wave
from dotenv import load_dotenv
import speech_recognition as sr
import sys
import subprocess
from gtts import gTTS

# load GPIO 
GPIO.setwarnings(False) # Ignore warnings for now 
GPIO.setmode(GPIO.BCM) # BCM vs Board 
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#load api key 
openai.api_key_path ='.env'

# record function 
# https://makersportal.com/blog/2018/8/23/recording-audio-on-the-raspberry-pi-with-python-and-a-usb-microphone 

form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 8192 #4096 # 2^12 samples for buffer
record_secs = 5 # seconds to record
dev_index = 1 # device index found by p.get_device_info_by_index(ii), changes this 
wav_output_filename = 'sound.wav' # name of .wav file

def recordAudio(): 
    audio = pyaudio.PyAudio() # create pyaudio instantiation

    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                        input_device_index = dev_index,input = True, \
                        frames_per_buffer=chunk)
    print("recording")
    frames = []

    # loop through stream and append audio chunks to frame array
    for ii in range(0,int((samp_rate/chunk)*record_secs)):
        data = stream.read(chunk)
        frames.append(data)

    print("finished recording")

    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save the audio frames as .wav file
    wavefile = wave.open(wav_output_filename,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()


def audioToGPT3():
    # read filename 
    filename = 'sound.wav'

    # initialize the recognizer
    r = sr.Recognizer()

    # open the file
    with sr.AudioFile(filename) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)
    
    # sent to gpt-3 
    response = openai.Completion.create(
        model="text-davinci-001",
        prompt=text, #replace with text from recording 
        temperature=0.4, #randomness 
        max_tokens=64, #charaters in response 
        top_p=1, #controls diversity
        frequency_penalty=0, #decrease repetition 
        presence_penalty=0 #increase likelihood to talk about new topic 
    )

    # response from gpt-3 
    result = response.choices[0].text
    print(result)

    # Passing the text and language to the engine 
    myobj = gTTS(text=result, lang='en', slow=False)
    
    # Saving the converted audio in a mp3 file named
    myobj.save("response.wav")

    # Playing the converted file
    os.system("mpg123 response.wav")

# function to run code when button is pressed 
def pressButton(channel): 
    #running record audio 
    recordAudio()

    # adding a sleep to make sure file has time to save before code runs 
    time.sleep(0.5)

    # running transcribe and send transcription to Gpt-3 and waiting for response 
    audioToGPT3()

# callback to execute function when button is pressed 
GPIO.add_event_detect(23,GPIO.FALLING, callback=pressButton, bouncetime=200)

# exit program 
message = input ("Press enter to quit")
GPIO.cleanup()

