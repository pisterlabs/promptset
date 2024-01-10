# pip install python-dotenv sounddevice numpy pydub pygame gtts wavio

import json
import os
import tkinter as tk
from tkinter import ttk
from dotenv import load_dotenv
import openai
import sounddevice as sd
from gtts import gTTS
import pygame
import wavio

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
openai.api_key = key


def set_wait_cursor():
    submit_btn.config(cursor="watch")
    app.update_idletasks()  # Force an immediate update of the window


def set_normal_cursor():
    submit_btn.config(cursor="")


def translate(language1, language2, text):
    set_label("Translating...")
    prompt = f"Translate the following from {language1} to {language2}: {text}"
    # prompt = 'Hi There'
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.6,
    )

    chat_gpt_translation = response["choices"][0]["message"]["content"]
    print('translation: ' + chat_gpt_translation)
    return chat_gpt_translation


def text_to_speech(translated_text, language):
    set_label("Playing...")
    tts = gTTS(translated_text, lang=languages[language], slow=False)
    tts.save('C:\\temp\\translation.mp3')
    # 5. Convert Translated Text to Speech
    # Placeholder for a TTS service (like Google Cloud TTS).

    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load('C:\\temp\\translation.mp3')
    pygame.mixer.music.play()


# If you want to keep the program running until the audio is done playing:
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # This will wait and let the music play.

    # close the mp3 file
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove('C:\\temp\\translation.mp3')


def capture_audio():
    # Indicate start of recording
    label_recording.config(text="Recording...", bg="red")

    app.update()
    # get number of seconds from dropdown
    duration = int(combo_duration.get())
    samplerate = 44100
    audio = sd.rec(int(samplerate * duration),
                   samplerate=samplerate, channels=2, dtype='int16')
    sd.wait()

    label_recording.config(text="Finished Recording", bg="green")

    app.update()

    # Save the numpy array to a WAV file using wavio
    wav_path = "c:\\temp\\myrecording.wav"
    wavio.write(wav_path, audio, samplerate, sampwidth=2)


def set_label(text):
    label_recording.config(text=text, bg="green", fg="white")
    label_recording.update()


def transcribe():
    audio_file = open("c:\\temp\\myrecording.wav", "rb")
    set_label("Transcribing...")
    transcription = openai.Audio.transcribe(model='whisper-1', file=audio_file)
    audio_file.close()
    print('transcription: ' + f'{transcription["text"]}\n\n')
    return transcription["text"]


def reset_status():
    label_recording.config(text="Click the button to start recording",
                           bg="lightgray", fg="white")
    label_recording.update()


def submit():

    set_wait_cursor()

    # 1. Capture Audio
    capture_audio()

    # 2. Transcribe the Audio
    transcription = transcribe()

    # translate the audio
    resulting_translation = translate(
        combo1.get(), combo2.get(), transcription)

    # 4. Translate the Text to speech
    text_to_speech(resulting_translation, combo2.get())

    reset_status()
    set_normal_cursor()


# create a dictionary to store the languages and their corresponding codes
languages = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Russian': 'ru',
    'Chinese (Simplified)': 'zh-CN',
    'Chinese (Traditional)': 'zh-TW',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Arabic': 'ar',
    'Dutch': 'nl',
    'Swedish': 'sv',
    'Turkish': 'tr',
    'Greek': 'el',
    'Hebrew': 'he',
    'Hindi': 'hi',
    'Indonesian': 'id',
    'Thai': 'th',
    'Filipino': 'tl',
    'Vietnamese': 'vi'
    # ... potentially more based on actual Whisper support
}


app = tk.Tk()
app.title("Babel Fish")
style = ttk.Style()

# Label and ComboBox for the first animal
label1 = ttk.Label(app, text="Select Known Language")
label1.grid(column=0, row=0, padx=10, pady=5)
combo1 = ttk.Combobox(
    app, values=list(languages.keys()))
combo1.grid(column=1, row=0, padx=10, pady=5)
combo1.set("English")

# Label and ComboBox for the second animal
label2 = ttk.Label(app, text="Select Translated Language:")
label2.grid(column=0, row=1, padx=10, pady=5)
combo2 = ttk.Combobox(
    app, values=list(languages.keys()))
combo2.grid(column=1, row=1, padx=10, pady=5)
combo2.set("Spanish")

label_recording_duration = tk.Label(app, text="Recording Duration (seconds):")
label_recording_duration.grid(column=0, row=2, padx=10, pady=5)
combo_duration = ttk.Combobox(app, values=[5, 10, 15, 20, 25, 30])
combo_duration.grid(column=1, row=2, padx=10, pady=5)
combo_duration.set(5)

# Button to submit the text to translate
submit_btn = ttk.Button(app, text="Record", command=submit)
submit_btn.grid(column=1, row=3, padx=10, pady=20)
label_recording = tk.Label(app, text="Click the button to start recording",
                           bg="lightgray", fg="white", width=60, height=2)
label_recording.grid(column=0, columnspan=2, row=8, padx=10, pady=20)

app.mainloop()
