import speech_recognition as sr
from gtts import gTTS
import os
import openai
from os import path
import soundfile as sf



openai.api_key = "****" #insert openAI API here
def recognize_hindi_speech(AUDIO_FILE):
    recognizer = sr.Recognizer()

    with sr.AudioFile(AUDIO_FILE) as source:
        audio = recognizer.record(source)

    # with sr.AudioFile(mp3_file_path.replace(".mp3", ".wav")) as source:
        # Save the pydub-loaded audio as a temporary WAV file
        # audio.export(source.name, format="wav")
        
        # print("Recognizing speech from MP3:")
        # recognizer.adjust_for_ambient_noise(source)  # Adjust for noise
        # audio = recognizer.record(source)  # Record the audio

    try:
        recognized_text = recognizer.recognize_google(audio, language="hi-IN")
        print("You said (Hindi):", recognized_text)
        return recognized_text
    except sr.UnknownValueError:
        print("Sorry, couldn't understand the speech.")
        return None
    except sr.RequestError as e:
        print("Error fetching speech recognition results:", e)
        return None
    
def chatgpt(userQues):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content":userQues}],
    )
    ans = response.choices[0].message.content
    return ans

def text_to_speech_hindi(text, output_file):
    max_characters = 1400  # Adjust as needed
    truncated_text = text[:max_characters]
    tts = gTTS(truncated_text, lang='hi')
    tts.save(output_file)


def main():
    mp3_file_path = 'Recording.mp3'

    wav_file_path = 'Recording.wav'


    data, samplerate = sf.read(mp3_file_path)
    sf.write(wav_file_path, data, samplerate)

    AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "Recording.wav")
    userQues=recognize_hindi_speech(AUDIO_FILE)
    
    ans=chatgpt(userQues)
    print(ans)
    
    with open('response.txt', 'w',encoding='utf-8') as f:
        f.write(ans)



    # print(ans)
    # if(ans==None):
    #     ans="Sorry, couldn't understand the speech."
    #     exit()
    # output_audio_file = "output_hindi.mp3"

    # text_to_speech_hindi(ans, output_audio_file)
    # play_audio(output_audio_file)

if __name__=="__main__":
    main()

