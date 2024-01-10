import speech_recognition as sr 
import datetime
import subprocess
import pywhatkit
import webbrowser
from gtts import gTTS
from pydub import AudioSegment
from playsound import playsound
import pyautogui
import time
import openai
import sched
import re
import threading

hot_words = ['jervis', 'jarvis']
audio_file_path = 'path/to/your/audio_file.wav'

api_key = 'YOUR_API_KEY'
openai.api_key = api_key

alarm_scheduler = sched.scheduler(time.time, time.sleep)
recognizer=sr.Recognizer()

def convert(text, input_filename='speech.mp3'):
    # Convert text to speech and save as MP3
    modified = text + ", my lord"
    tts = gTTS(modified)
    tts.save(input_filename)

    # Load the MP3 file
    audio = AudioSegment.from_mp3(input_filename)

    # Speed up the audio by 2x during export
    #modified_audio = audio.speedup(playback_speed=1.5)

    #modified_audio.export(input_filename, format="mp3")
    audio.export(input_filename, format="mp3")
    playsound(input_filename)

def play_music(query):
    pywhatkit.playonyt(query)
    time.sleep(5)  # Adjust the sleep duration based on your system's performance and network speed
    pyautogui.press('space')  # Press the spacebar to play/pause the video
    print(query)

def play_alarm(sound_file):
    t_end = time.time() + 60 * 3
    while time.time() < t_end:
        print("Wake up!")
        playsound(sound_file)

def set_alarm(hour, minute):
    # Get the current time
    current_time = time.localtime()
    current_hour, current_minute = current_time.tm_hour, current_time.tm_min

    # Calculate the time difference in seconds until the alarm
    time_difference = (hour - current_hour) * 3600 + (minute - current_minute) * 60

    # Check if the specified time is in the future
    if time_difference > 0:
        print(f"Alarm set for {hour:02d}:{minute:02d}")
        
        # Schedule the alarm

        # Create a timer for the alarm
        alarm_timer = threading.Timer(time_difference, play_alarm, args=('alarm.wav',))
        #alarm_thread = threading.Thread(target=play_alarm)
        alarm_timer.start()
        #alarm_scheduler.enter(time_difference, 1, play_alarm, argument=('alarm.wav',))
        #alarm_scheduler.run()
    else:
        print("Please set an alarm for a future time.")

def chat_with_assistant(chat):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "you are a personal assistant called jarvis, you manage my well-being and keep track of my bedtime. You may call me my lord. I normally wake up at 8am and go to bed at 12 am mid night.you are also responsible for my emotional well-being. Pretend to sound like medival women, use vocabularly from medival time"},
            {"role": "user", "content": chat},
        ],

        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    assistant_response = response['choices'][0]['message']['content']
    print("Assistant:", assistant_response)

    convert(assistant_response)

def cmd():

    with sr.Microphone() as source:
        print("Clearing background noises...Please wait")
        recognizer.adjust_for_ambient_noise(source,duration=0.5)
        print('Ask me anything..')
        recordedaudio=recognizer.listen(source)

    try:
        text=recognizer.recognize_google(recordedaudio,language='en_US')
        text=text.lower()
        print('Your message:',format(text))

        if any(word in text for word in hot_words):  
            if 'firefox' in text:
                c='Opening firefox..'
                convert(c)
                programName = "/usr/bin/firefox"
                subprocess.Popen([programName])

            elif 'time' in text:
                time = datetime.datetime.now().strftime('%I:%M %p')
                print(time)
                convert(time)
                #chat_with_assistant(time)

            elif 'play' in text:
                omit_words = ['jarvis', 'jervis', 'play']
                words = text.split()
                filtered_words = [word for word in words if word.lower() not in omit_words]
                filtered_text = ' '.join(filtered_words)
                filtered_text_ = 'playing'+filtered_text
                convert(filtered_text_)
                play_music(text)
                

            elif 'set alarm' in text:
                match = re.search(r'\b(\d{1,2}:\d{2})\b', text)
                if match:
                    time_str = match.group(1)
                    hour, minute = map(int, time_str.split(':'))
                    print(f"Hour: {hour}, Minute: {minute}")
                    set_alarm(hour,minute)
                    
                else:
                    print("Time not found in the text.")
                
            elif 'youtube' in text:
                b='opening youtube'
                convert(b)
                webbrowser.open('www.youtube.com')
            
            elif '' in text:
                chat_with_assistant(text)
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    while True:
        cmd()
















