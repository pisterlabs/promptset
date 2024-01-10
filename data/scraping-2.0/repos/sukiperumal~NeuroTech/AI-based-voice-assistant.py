import openai
import speech_recognition as sr
from gtts import gTTS
import os

# Set your OpenAI API key
openai.api_key = "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"

# Initialize the recognizer
recognizer = sr.Recognizer()

# Function to listen to the user's voice input
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        print("Processing...")

    try:
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition; {0}".format(e))
        return None

# Function to speak a response
def speak(response):
    tts = gTTS(response)
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")  # Adjust the player command as needed

# Main loop for the voice assistant
while True:
    user_input = listen()
    
    if user_input:
        # Send user's voice input to ChatGPT
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=user_input,
            max_tokens=50  # Adjust as needed
        )

        chatgpt_response = response.choices[0].text
        print("ChatGPT says: " + chatgpt_response)

        speak(chatgpt_response)