import openai
from key import secret_key

import streamlit as st
from streamlit_chat import message
import base64

import speech_recognition as sr
from textblob import TextBlob

from gtts import gTTS
import os
from playsound import playsound

openai.api_key = secret_key
language ="en"

def add_image(filepath):

    #this method allows image to occupy the entire sidebar
    image_extension = "jpg"

    # opens file in binary format for reading
    with open(filepath, "rb") as f: 
        data = f.read()
        # base64 allows data to be encoded into a string, and then embedded directly into css code
        encoded = base64.b64encode(data)

    # insert encoded file into the HTML element, change the "background-image" property
    sidebar_image = f"""
    <style>
    [data-testid="stSidebar"]> div:first-child{{
    background-image: url(data:image/{"image_extension"};base64,{encoded.decode()});
    background-size: cover;
    }}
    </style>
    """

    st.markdown(sidebar_image,unsafe_allow_html=True)
        
    # Image by "https://www.freepik.com/free-vector/flat-lovely-spring-landscape-background_12126865.htm#query=rural%20landscape&position=28&from_view=keyword"



class Sheep:
    def __init__(self, r, w):
        self.mood = "smiling"
        self.polarity = 0
        self.last_received = ""
        self.last_replied = ""
        self.placeholder = st.empty()

    def reply(self, message):
        self.last_received = message
        # use GPT-3's text completion

        # parameters:
        # temperature: 0 ~ 1. 1 means more random
        # max_tokens: max words in response
        # top_p: top % of results to consider
        # frequency_penalty: -2 ~ 2. 2: generate messages different from previous messages
        # presence_penalty: -2 ~ 2. 2: generate text different from ones in the same message
        reply = openai.Completion.create(engine = "text-davinci-003",prompt=message, temperature = 0.4,max_tokens=300,top_p=1.0,frequency_penalty=0.5,presence_penalty=0.0, n=1, stop = None)
        reply_str = reply.choices[0].text
        self.latest = reply_str

        return reply_str

    def speak(self):
        tts = gTTS(text=self.latest,lang=language,slow=False)
        tts.save("temp.mp3")
        playsound("temp.mp3")
        os.remove("temp.mp3")
    
    def detect_emotion(self):
        text_block = TextBlob(self.last_received)
        self.polarity = text_block.sentiment.polarity
        if self.polarity > 0.5:
            new_mood = "happy"
        elif self.polarity > -0.1:
            new_mood = "smiling"
        else:
            new_mood = "unhappy"
        self.mood = new_mood

    def display_expression(self):
        self.placeholder.empty()
        with st.sidebar:
            self.placeholder = st.image("graphics/"+str(self.mood)+".gif")



# set up speech recognition
r = sr.Recognizer()
mic = sr.Microphone(device_index=1) # 0 ~ num devices-1. If there is an error, use the following two lines to help adjust index

# print(sr.Microphone.list_microphone_names())
# print(sr.Microphone(device_index=1))
source = mic


# webapp design from top to bottom using streamlit. 
# script is run every time the user changes a widget, e.g. clicks on a button, hits enter in a textbox
st.sidebar.title("ChatShirley")
with st.sidebar:
    mode = st.radio("You:",("Text", "Voice"))

if mode=="Text":
    with st.sidebar:
        user_input = st.text_input("", "",key="user_box")
    
if mode=="Voice":
    user_input = ""
    with st.sidebar:
        talk_button = st.button("🎤")
        listening_indicator = st.empty()
            
    # if user clicks the talk button
    if talk_button:
        listening_indicator.write("listening...")
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)
            listening_indicator.write("No longer listening.")
        try:
            user_input = r.recognize_google(audio)
        except:
            pass
        listening_indicator.empty()

Shirley = Sheep(0.4, 300)
Shirley.display_expression()

add_image("graphics/mountainside.jpg")


# session_state is a dict-like object to store variables across sessions/different runs
if "inputs" not in st.session_state:
    st.session_state["inputs"] = []

if "outputs" not in st.session_state:
    st.session_state["outputs"] = []


# if the user_input is not empty
if user_input:
    output = Shirley.reply(user_input)
    Shirley.detect_emotion()
    Shirley.display_expression()

    # store new messages
    st.session_state.inputs.append(user_input)
    st.session_state.outputs.append(output)


# display all messages
if st.session_state["outputs"]:
    for i in range(0, len(st.session_state['inputs']), 1):
        # most recent messages at the bottom
        message(st.session_state['inputs'][i], is_user=True, key=str(i)+"user", avatar_style="identicon", seed=2)
        message(st.session_state["outputs"][i], avatar_style="identicon",key=str(i),seed=11)

#st.write(Shirley.polarity)
#st.write(Shirley.mood)

# only speak after all messages are displayed
if user_input:
    Shirley.speak()