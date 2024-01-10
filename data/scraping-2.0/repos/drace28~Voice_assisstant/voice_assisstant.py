import speech_recognition as sr
import pyttsx3
import openai
import pyautogui
import os
import time
from dotenv import load_dotenv
import psutil
import google.generativeai as genai
from pynput.keyboard import Controller, Key

# Load the .env file
load_dotenv()
keyboard = Controller()

# Set your API keys
openai.api_key = os.getenv('OPENAI_KEY')
GOOGLE_API_KEY= os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gimini-pro")

def is_spotify_running():
    for process in psutil.process_iter(['name']):
        if process.info['name'] == 'Spotify.exe':
            return True
    return False

# Initialize the recognizer
r = sr.Recognizer()

# Function to convert text to speech
def SpeakText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

# Function to check if the text contains the wake word
def wakeWord(text):
    WAKE_WORDS = ['assistant', 'hey anisha', 'anisha', 'okay anisha', 'hi anisha', 'hello anisha']
    text = text.lower()
    return any(word in text for word in WAKE_WORDS)


# Function to perform actions based on user commands
def performAction(command):
    if "open" in command:
        words = command.split()
        index = words.index("open")
        program = words[index + 1]
        SpeakText(f"Opening{program}")
        os.system(f'start {program}')
    elif "close" in command:
        words = command.split()
        index = words.index("close")
        program = words[index + 1]
        SpeakText(f"Closing{program}")
        os.system(f'taskkill /f /im {program}.exe')
    elif "type" in command:
        words = command.split()
        index = words.index("type")
        text = words[index + 1:]
        text = " ".join(text)
        SpeakText(f"Typing{text}")
        pyautogui.typewrite(text)
    elif "search" in command:
        words = command.split()
        index = words.index("search")
        query = words[index + 1:]
        query = " ".join(query)
        SpeakText(f"Searching for {query}")
        os.system(f'start https://www.google.com/search?q={query}')
        try:
        # Send the user's question to ChatGPT
            response = openai.Completion.create(
                engine="davinci",
                prompt=f"I am searching for {query}",
                temperature=0.7,
                max_tokens=150,
                n=1,
            )
            SpeakText(response["choices"][0]["text"].strip())
        except Exception as e:
            print(f"Error querying ChatGPT: {e}")
            SpeakText("I'm sorry, I couldn't generate a response at the moment.")
    elif "send whatsapp message" in command:
        # Extract the recipient and message from the command
        recipient, message = command.replace("send whatsapp message ", "").split(" message ")
        SpeakText(f"Sending WhatsApp message to {recipient}")
        # Open WhatsApp
        os.system('start whatsapp')  # Replace with the path to your WhatsApp.exe
        time.sleep(5)  # Wait for WhatsApp to open
        # Click on the search box
        pyautogui.click(x=110, y=200)  # Replace with the coordinates of your search box
        # Type the recipient's name and press enter
        pyautogui.write(recipient)
        pyautogui.press('enter')
        time.sleep(2)  # Wait for the chat to open
        # Type the message and press enter
        pyautogui.write(message)
        pyautogui.press('enter')
    
    elif "play music" in command or "pause music" in command or "next track" in command or "previous track" in command:
        if not is_spotify_running():
            SpeakText("Opening Spotify")
            os.system('start spotify')  # This command opens Spotify on Windows
        if "play music" in command or "pause music" in command:
            SpeakText("OK")
            with keyboard.pressed(Key.media_play_pause):
                pass  # This shortcut plays/pauses music in Spotify
        elif "next track" in command:
            SpeakText("Skipping to the next track on Spotify")
            with keyboard.pressed(Key.media_next):
                pass  # This shortcut skips to the next track in Spotify
        elif "previous track" in command:
            SpeakText("Going to the previous track on Spotify")
            with keyboard.pressed(Key.media_previous):
                pass
    
    elif "ask a question" or "bard" in command:
        SpeakText("Sure, what would you like to ask?")
        audio_data = r.listen(source, timeout=10)  # Listen for the user's question
        question = r.recognize_google(audio_data, language='en-US')
        response = model.generate_content(question)
        SpeakText(response)
    elif "bye" in command:
        SpeakText("Bye, See you soon")
        exit()
    elif "goodbye" in command:
        SpeakText("Goodbye, See you soon")
        exit()
    elif "good night" in command:
        SpeakText("Good Night, Sweet Dreams")
        exit()
    else:
        SpeakText("Sorry, I did not understand that.")

# Function to query ChatGPT
def chatgpt_query(question):
    try:
        # Send the user's question to ChatGPT
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=question,
            temperature=0.7,
            max_tokens=150,
            n=1,
        )

        return response["choices"][0]["text"].strip()

    except Exception as e:
        print(f"Error querying ChatGPT: {e}")
        return "I'm sorry, I couldn't generate a response at the moment."

# Continuous listening loop
while True:
    print("Say something")

    with sr.Microphone(device_index=1) as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        audio_data = r.listen(source)
        print("Recognizing...")

        try:
            MyText = r.recognize_google(audio_data, language='en-US')
            print(MyText)

            # if "bye" or "goodbye" or "goodnight" in command:
            #     SpeakText("Bye, have a good day")
            #     exit()


            if wakeWord(MyText):
                SpeakText("Hello, How can I assist you?")

                # Listen for the user's command after the wake word
                audio_data = r.listen(source, timeout=5)
                command = r.recognize_google(audio_data, language='en-US')

                # Perform actions based on the user's command
                performAction(command)
            
            elif "bye" or "goodbye" or "goodnight" in command:
                SpeakText("Bye, have a good day")
                exit()

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Error with the speech recognition service; {e}")
        except Exception as e:
            print(f"An error occurred; {e}")
