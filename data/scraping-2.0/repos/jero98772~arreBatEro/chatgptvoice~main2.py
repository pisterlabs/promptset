import speech_recognition as sr
import openai
from gtts import gTTS
from pygame import mixer
import time
 
# Set up OpenAI API credentials
openai.api_key = ""
 
# Configure TTS engine
#tts_engine = pyttsx3.init()
 
# Configure speech recognition engine
r = sr.Recognizer()
 
# Conversation loop
while True:
    # Listen for user speech input
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)  # Optional: Adjust for ambient noise levels
        audio = r.listen(source)
 
    # Convert speech to text
    print('finished listening')
    user_input = r.recognize_google(audio)
 
 
    # Send user input to ChatGPT
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=user_input,
        max_tokens=50
    )
 
    # Get ChatGPT's response
    chat_response = response.choices[0].text.strip()
    print('Chatgpt response ready')
    # Speak ChatGPT's response
    tts = gTTS(text=chat_response, lang='en')
    tts.save('audio.mp3')
 
     # Play the audio
    mixer.init()
    mixer.music.load("audio.mp3")
    mixer.music.play()
 
    # Wait for the audio to finish playing
    while mixer.music.get_busy():
        time.sleep(0.1)
 
 
    # Stop conversation loop if exit command is given
    if 'exit' in user_input:
        break
