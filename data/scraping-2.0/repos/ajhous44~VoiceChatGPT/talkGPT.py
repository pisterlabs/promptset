import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import os
from dotenv import load_dotenv

# Initialize recognizer
r = sr.Recognizer()

# Set up OpenAI API credentials
load_dotenv()  # load variables from .env
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_response(prompt):
    
    """
    Generate a response using OpenAI API
    """
    try:
        # FOR USE WITH CHAT COMPLETION API + GPT-3.5-TURBO
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": prompt}]
        )
        print("Tokens used:", completion.usage.total_tokens)
        return completion.choices[0].message.content
        
        # FOR USE WITH COMPLETION API + DAVINCI
        # response = openai.Completion.create(
        #     engine='text-davinci-003', 
        #     prompt=prompt,
        #     max_tokens=100,
        #     temperature=0.3,
        # )
        # return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error in generating response: {e}")
        return None

def play_audio(file_path):
    """
    Play audio from a file
    """
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

    pygame.mixer.music.unload()

def create_audio(text, filename):
    """
    Create an audio file from text
    """
    try:
        speech = gTTS(text=text, lang='en', slow=False)
        speech.save(filename)
    except Exception as e:
        print(f"Error in creating audio: {e}")

# Use the microphone as source for input.
with sr.Microphone() as source:
    # Play an introduction audio
    create_audio("At your service!", "response.mp3")
    play_audio("response.mp3")

    context = "Be sure to ask follow up questions if applicable."
    while True:
        print("Listening...")
        # Read the audio data from the default microphone
        try:
            audio_data = r.listen(source, timeout=1)
            print("Recognizing...")

            # Convert speech to text
            text = r.recognize_google(audio_data)
            print(f"User: {text}")

            # Exit condition
            if 'exit' in text or 'stop' in text:
                create_audio("It was great chatting with you!", "response.mp3")
                play_audio("response.mp3")
                print("Exiting...")
                break

            else:
                # Generate AI response
                response_text = generate_response(context + text)
                print('AI:', response_text)

                # Create TTS audio and play it
                if response_text:
                    create_audio(response_text, "response.mp3")
                    play_audio("response.mp3")

        except sr.WaitTimeoutError:
            print('Timeout: No speech detected for 3 seconds.')
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio or you're not talking.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
