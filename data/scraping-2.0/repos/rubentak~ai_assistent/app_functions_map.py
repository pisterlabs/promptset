#%%
import folium
from gtts import gTTS
import pygame
import tempfile
import pyaudio
import wave
import time
from audio_get_channels import get_cur_mic
import openai
import credentials
import os
from geopy.geocoders import Nominatim
import json
from urllib.request import urlopen
import pandas as pd
from geo_google import get_surroundings


'''these functions are used to run the chatbot and audio functions in the flask app'''

script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "audio_output.wav")
preprompt = "You are a Ai audio guide. in the following prompt, look for the name of a city or a location, and give a one line discription of this place. Start with the location as a header."


#%%   AUDIO FUNCTIONS
def audio_spectrum(num_seconds):
    ''' Record audio for num_seconds and save it to a wav file'''

    script_dir = os.path.dirname(os.path.abspath(__file__))

    chunk = 2205
    channels = 1
    fs = 44100
    seconds = max(num_seconds, 0.1)
    sample_format = pyaudio.paInt16
    filename = os.path.join(script_dir, "audio_output.wav")

    print(f'\n... Recording {seconds} seconds of audio initialized ...\n')

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    input_device_index=get_cur_mic(),
                    frames_per_buffer=chunk,
                    input=True)


    frames = []
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk, False)
        frames.append(data)


    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

#%%   TRANSCRIPT FUNCTIONS
def get_transcript_whisper():
    '''Get the transcript of the audio file'''
    openai.api_key = credentials.api_key
    file = open(filename, "rb")
    transcription = openai.Audio.transcribe("whisper-1", file, response_format="json")
    transcribed_text = transcription["text"]

    return transcribed_text

#%%  TEXT RECOGNITION FUNCTIONS
def recognize_infos(transcribed_text):
    ''' If transcribed text contains a city name from worldcities.csv, return the city name'''
    df = pd.read_csv('worldcities.csv')
    df = df.dropna()
    df = df.drop_duplicates(subset=['city_ascii'])
    df = df.reset_index(drop=True)
    df['city_ascii'] = df['city_ascii'].str.lower()

    # Convert transcribed_text to lowercase for proper comparison
    transcribed_text = transcribed_text.lower()

    for city in df['city_ascii']:
        if city in transcribed_text:
            city = df[df['city'] == city.lower()]['city'].values[0]
            lng = df[df['city_ascii'] == city.lower()]['lng'].values[0]
            lat = df[df['city_ascii'] == city.lower()]['lat'].values[0]
            location = {'lat': lat, 'lng': lng}
            return city, location


#%%   CHATGPT FUNCTIONS
def run_chatGPT(prompt):
    '''Run chatGPT with the prompt and return the response'''
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": preprompt + prompt},
        ]
    )
    answer = completion.choices[0].message.content

    return answer

#%%   SPEAK FUNCTIONS
def speak_answer(answer):
    '''Speak the answer'''
    tts = gTTS(text=answer, lang='en')
    with tempfile.NamedTemporaryFile(delete=True) as f:
        tts.save(f.name)
        pygame.mixer.music.load(f.name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


#%%    CREATE MAP
#load map in browser
def create_map(location, df):
    map = folium.Map(location=[location['lat'], location['lng']], zoom_start=15, tiles='Stamen Toner')
    for i in range(0, len(df)):
        button_html = f'<button onclick="location.href=\'/speak?text={df.iloc[i]["name"]}\';">Story</button>'
        marker_popup = folium.Popup(f'''
            Name: {df.iloc[i]['name']}
            Vicinity: {df.iloc[i]['vicinity']}
            Business status: {df.iloc[i]['business_status']}
            Rating: {df.iloc[i]['rating']}
            User ratings total: {df.iloc[i]['user_ratings_total']}
            Opening hours: {df.iloc[i]['opening_hours']} 
            {button_html}
            ''')
        folium.Marker([df.iloc[i]['lat'], df.iloc[i]['lng']], popup=marker_popup,
                      icon=folium.Icon(color='red', icon='')).add_to(map)
    return map._repr_html_()


def run_all_functions():
    try:
        audio_spectrum(6)
    except KeyboardInterrupt:
        pass

    transcript = get_transcript_whisper()
    recognized_city = recognize_infos(transcript)
    city = recognized_city[0]
    location = recognized_city[1]
    df = get_surroundings(location['lat'], location['lng'], 500, 'tourist_attraction', '')
    map = create_map(location, df)


    # If text contains one word of a stopwordlist then the script will stop
    if any(word in transcript for word in ['stop', 'Stop', 'exit', 'quit', 'end']):
        print('... Script stopped by user')
        exit()

    transcript = f' {transcript}'

    answer = run_chatGPT(transcript)
    speak_answer(answer)

    return transcript, answer, city, map


#run_all_functions()
# ----------------------------------------------------------------
if __name__ == "__main__":
    script_start = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, "audio_output.wav")
    run_all_functions()

