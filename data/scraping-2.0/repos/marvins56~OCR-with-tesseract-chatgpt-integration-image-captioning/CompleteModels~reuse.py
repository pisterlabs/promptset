import openai

import datetime
import cv2
import pytesseract

import speech_recognition as sr

import datetime  # required to resolve any query regarding date and time
import speech_recognition as sr  # required to return a string output by taking microphone input from the user
import pyttsx3
# import wikipedia  # required to resolve any query regarding wikipedia
import webbrowser  # required to open the prompted application in web browser
import os.path  # required to fetch the contents from the specified folder/directory
import smtplib  # required to work with queries regarding e-mail
from model import get_caption_model, generate_caption
import os

# Connects pytesseract(wrapper) to the trained tesseract module
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR'
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
openai.api_key = "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"


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 50)  # Decrease the rate by 50

# print(voices[1].id)
engine.setProperty('voice', voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def wishMe():
    hour = int(datetime.datetime.now().hour)
    # hour = datetime.datetime.now().hour
    if(hour >= 6) and (hour < 12):
        speak(f"Good Morning ")
    elif(hour >= 12) and (hour < 18):
        speak(f"Good afternoon ")
    elif(hour >= 18) and (hour < 21):
        speak(f"Good Evening ")
def capture_image():
    speak("note : you will be asked to retake the image incase the image is not clear.")
    speak("press button to capture image in , 1 , 2 , 3. ")
    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Wait for user to press a key to take image
    while True:
        ret, frame = cap.read()
        cv2.imshow('Press Space to Capture Image', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Get current date and time
            now = datetime.datetime.now()
            filename = now.strftime("%Y-%m-%d_%H-%M-%S")
            # Create directory if it does not exist
            directory = './images/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Save image
            filepath = os.path.join(directory, f"{filename}.png")
            cv2.imwrite(filepath, frame)
            # Release camera and close window
            cap.release()
            cv2.destroyAllWindows()
            return filepath

def speech_to_text():
    # Initialize speech recognizer
    r = sr.Recognizer()
    # Use default system microphone as source to listen to speech
    with sr.Microphone() as source:
        speak("hello, welcome.... How may i be of service....")
        # Adjust for ambient noise
        r.adjust_for_ambient_noise(source)
        # Record the user's speech
        audio = r.listen(source)
    try:
        # Use Google speech recognition to convert speech to text
        text = r.recognize_google(audio)
        speak(f"You said: {text}")
        speak("Noted.")
        return text

    except sr.UnknownValueError:
        speak("Sorry, could not understand your input.")
    except sr.RequestError:
        speak("Sorry, there was an error with the speech recognition service.")

        # Return empty string on error
        return ""


def generate_response(prompt):
    # Generate response from OpenAI API
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    # Extract the response text
    message = response.choices[0].text.strip()
    return message


def CaptionImage(image_path):
    try:
        speak("Generating caption.....")
        caption_model = get_caption_model()
        captions = []
        pred_caption = generate_caption(image_path, caption_model)

        captions.append(pred_caption)
        speak("Generation complete ..")
        for _ in range(4):
            pred_caption = generate_caption(image_path, caption_model, add_noise=True)
            if pred_caption not in captions:
                captions.append(pred_caption)

        speak("Processing complete. The generated captions are:")
        for c in captions:
            speak(c)
    except FileNotFoundError:
        speak("Error: could not find the vocabulary file.")