"""
pip install PyQt5==5.15.4
pip install PyQt5-sip==12.9.0
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import speech_recognition as sr
import openai
from tensorflow import keras
import nltk
from nltk.stem import WordNetLemmatizer
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
# from PyQt6.QtGui import QPixmap
# from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QMovie
import pyttsx3
import pickle
import threading
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


lemmatizer = WordNetLemmatizer()
with open(r'intents.json', encoding='utf-8') as file:
    intents = json.load(file)
# Load words and labels from files
with open("words.pkl", "rb") as file:
    words = pickle.load(file)

with open("labels.pkl", "rb") as file:
    labels = pickle.load(file)
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)
# Load the trained model instead of retraining it
model = keras.models.load_model('model.h5')
api_key = "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"
openai.api_key = api_key

model_engine = "gpt-3.5-turbo"
patterns = []
for intent in intents['intents']:
    for phrase in intent['patterns']:
        patterns.append(phrase)


def record_and_transcribe_audio(lang):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        if lang == "en-US":
            transcript = recognizer.recognize_google(audio, language="en-US")
        else:
            transcript = recognizer.recognize_google(audio, language="fr-FR")
        print(f"Transcribed text: {transcript}")
        return transcript
    except sr.UnknownValueError:
        print("Could not understand audio. Please try again.")
        return ""





def play_text(text, lang='fr'):
    # Initialize pyttsx3 engine
    engine = pyttsx3.init()

    voices = engine.getProperty('voices')
    french_voice = None
    english_voice = None

    for voice in voices:
        if 'french' in voice.name.lower():
            french_voice = voice
            print(f"Voice name: {voice.name}, Voice ID: {voice.id}")  # Print available French voices
        if 'english' in voice.name.lower():
            english_voice = voice

    if lang == 'fr' and french_voice:
        engine.setProperty('voice', french_voice.id)
    elif lang == 'en' and english_voice:
        engine.setProperty('voice', english_voice.id)
    # Convert text to speech
    engine.say(text)
    engine.runAndWait()


def calculate_similarity(text, patterns):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([text] + patterns)
    similarity = cosine_similarity(matrix)[0, 1:]
    return similarity

def get_most_similar(text, patterns):
    similarity = calculate_similarity(text, patterns)
    most_similar_index = np.argmax(similarity)
    return patterns[most_similar_index]

def predict_response(text):
    pattern_bag = bag_of_words(text, words)
    pattern_bag = pattern_bag.reshape(1, -1)
    prediction = model.predict(pattern_bag)
    prediction_index = np.argmax(prediction)
    return labels[prediction_index]

def chatbot_response(text, asr_language):
    if text.lower() == 'stop':
        return True, ''
    most_similar_pattern = get_most_similar(text, patterns)
    similarity = calculate_similarity(text, [most_similar_pattern])[0]
    print(similarity)
    if similarity > 0.5:
        response = predict_response(text)
        return False, response
    else:
        completion = openai.Completion.create(
            engine="text-davinci-003",
            prompt=text,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.3,
        )
        # sorry = ''
        # sorryen = "Sorry Emines team did not train me on this question, but here is my personal take on it"
        # sorryfr = "Désolé, je n'ai pas été entraîné sur cette question par Emines, mais voici ce que j en pense"
        # if asr_language == 'en-US':
        #     sorry = sorryen
        # else :
        #     sorry = sorryfr
   
        return False, completion.choices[0].text

# stop_event = threading.Event()
def run_chatbot(selected_lang):
    asr_language = 'en-US' if selected_lang.lower() == 'english' else 'fr-FR'

    if selected_lang.lower() == 'french':
        play_text("Bonjour?", lang='fr')
    else:
        play_text("Hey?", lang='en')

    while True:
        text = record_and_transcribe_audio(asr_language)
        if not text:
            print("Could not understand audio. Please try again.")
            continue

        stop, response = chatbot_response(text,asr_language)
        if stop  :
            break
        if selected_lang.lower() == 'french':
            play_text(response, lang='fr')
        else:
            play_text(response, lang='en')

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')
Config.set('graphics', 'resizable', '1')
Config.set('graphics', 'width', '800')
Config.set('graphics', 'height', '600')
Config.write()

class ChatbotWindow(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        # Sleepy gif
        self.sleepy_movie = Image(source='giphy.gif')
        self.add_widget(self.sleepy_movie)

        # English button
        self.english_button = Button(text='English')
        self.english_button.size_hint = (1, 0.1)
        self.add_widget(self.english_button)
        self.english_button.bind(on_press=self.start_english_chatbot)

        # French button
        self.french_button = Button(text='French')
        self.french_button.size_hint = (1, 0.1)
        self.add_widget(self.french_button)
        self.french_button.bind(on_press=self.start_french_chatbot)

        # Stop button
        self.stop_button = Button(text='Sleep')
        self.stop_button.size_hint = (1, 0.1)
        self.add_widget(self.stop_button)
        self.stop_button.bind(on_press=self.reset_app)

    def reset_app(self, *args):

        # Remove the current sleepy_movie widget
        self.remove_widget(self.sleepy_movie)

        # Add a new sleepy_movie widget
        self.sleepy_movie = Image(source='giphy.gif')
        self.add_widget(self.sleepy_movie)

        # Insert the new widget at the same position as the initial image in the widget tree
        # self.children.insert(len(self.children) - 1, self.sleepy_movie)

        # Ensure that the buttons stay in their original position
        self.remove_widget(self.english_button)
        self.remove_widget(self.french_button)
        self.remove_widget(self.stop_button)

        self.add_widget(self.english_button)
        self.add_widget(self.french_button)
        self.add_widget(self.stop_button)
    def start_english_chatbot(self, *args):
        self.sleepy_movie.source = 'giphy2.gif'
        self.sleepy_movie.reload()
        self.run_chatbot('english')

    def start_french_chatbot(self, *args):
        self.sleepy_movie.source = 'giphy2.gif'
        self.sleepy_movie.reload()
        self.run_chatbot('french')

    def run_chatbot(self, selected_lang):
        my_thread = threading.Thread(target=run_chatbot, args=(selected_lang,))
        my_thread.start()


class ChatbotApp(App):
    def build(self):
        return ChatbotWindow()

if __name__ == '__main__':
    ChatbotApp().run()


