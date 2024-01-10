from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
import os
import pyttsx3
import speech_recognition as sr
import threading
import time
from gtts import gTTS
import playsound
from streamlit_card import card
import base64
from streamlit_option_menu import option_menu
import requests
from PIL import Image
from io import BytesIO

from streamlit_card import card

import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random

# restore all of our data structures
import pickle
import requests
from tensorflow.keras.models import load_model

api_url="http://127.0.0.1:8000/fetchanswer/"
stemmer = LancasterStemmer()

data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('IntentBotData1.json') as json_data:
    intents = json.load(json_data)

# load our saved model
model = load_model("Intent_Model.h5")

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

ERROR_THRESHOLD = 0.55
def classify(sentence):
    # generate probabilities from the model
    results = model.predict(np.array([bow(sentence, words)]))[0]
    print(results)
    print(len(results))
    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    print(results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    print(results)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    print(return_list)
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return random.choice(i['responses'])
            results.pop(0)
    else:
        data = None  # Initialize data with None

        try:
            params = {"question": sentence}
            response = requests.get(api_url, params=params)
            print(response.text)
            data = response.json()
        except Exception as e:
            print(e)

        # Now you can check if data is not None before trying to access it
        if data is not None:
            return data['question']
        else:
            return "An error occurred"



PAGE_TITLE: str = "AI Talks"
PAGE_ICON: str = "🤖"
LANG_EN: str = "En"

location_map=['washroom']

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
st.title('🤖 Parklink System Chatbot')

selected_lang = option_menu(
    menu_title="",
    options=[LANG_EN],
    icons=["globe2", "translate"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)
# Store the image URL in session state
if 'image_url' not in st.session_state:
    st.session_state['image_url'] = "url"

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []


# Create a pyttsx3 object for text to speech
engine = pyttsx3.init()

# Create a speech recognition object for speech to text
r = sr.Recognizer()

# Define a function to listen to the user's voice and return the text
def listen():
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        # Listen for the user's voice
        print("Listening...")
        audio = r.listen(source)
    try:
        # Recognize the user's voice using Google Speech Recognition
        text = r.recognize_google(audio)
        # Print the user's input
        print(f"You said: {text}")
        # Return the text
        return text
    except Exception as e:
        # If there is an error, return None
        print(str(e))
        print("Sorry, I could not understand you.")
        return None

# Define a function to speak the chatbot's response
def speak(text):
    try:
        print(f"AI said: {text}")
        tts = gTTS(text=text, lang='en',tld='ca')
        tts.save("good.mp3")
        playsound.playsound('good.mp3')
        os.remove('good.mp3')    
    except Exception as e:
        print("Exception occured:",e)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

if st.button("Voice-Search 🎤"):
# Listen to the user's input and store it in a variable
    query = listen()
    if query:
        with st.spinner("processing..."):
            result=response(query)
            print("Response------>",result)
            # conversation_string = get_conversation_string()
            # # st.code(conversation_string)
            # refined_query = query_refiner(conversation_string, query)
            # st.subheader("Refined Query:")
            # st.write(result)
            # context = find_match(refined_query)
            # print(context)  
            # response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            # if "content/images" in response and "jpg" in response:
            #     speak("Here is the map for the desired location:")
            # else:
            response=result
            speak(result)
            # print("hello")
            
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 


     

with textcontainer:
    with st.form("Query Form", clear_on_submit=True):
        query = st.text_input("Query: ", key="input")
        submit_button = st.form_submit_button("Submit")

        if submit_button and query:
            with st.spinner("typing..."):
                result = response(query)
                print("Response------>", result)
                response = result
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)


with response_container:
    print(st.session_state['responses'])
    print(response_container)
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            if "images/" in st.session_state['responses'][i] and st.session_state['image_url']!="url":
                st.image(st.session_state['responses'][i], caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
            
        
        try:
            print(st.session_state['responses'])
            last_element=st.session_state['responses'][-1]
            if 'images/' in last_element and '.jpeg' in last_element and st.session_state['image_url']=="url":                
                print("True")
                with st.chat_message("Response:"):
                    contentIndex=st.session_state['responses'][-1].index('images')
                    jpgIndex=st.session_state['responses'][-1].index('.jpeg')
                    url=st.session_state['responses'][-1][contentIndex:jpgIndex+5]
                    print(st.session_state['responses'][-1][contentIndex:jpgIndex+5])
                    print(st.session_state['image_url'])
                    st.session_state['image_url'] = url
                    print(st.session_state['image_url'])
                    st.image(st.session_state['image_url'], caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
         
        except Exception as e:
            
            print("Exception Occured :", e)
          