# This script was written by Ibrahim Al-Shinnawi, shinnawi.com, on 2024-01-06.

import os
import speech_recognition as sr
from openai import OpenAI

# Initialize OpenAI client with API key from environment variable 'OPENAI_API_KEY'
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize the speech recognition recognizer
recognizer = sr.Recognizer()

# Set up the microphone for audio input
microphone = sr.Microphone()

def ask_chatgpt(question):
    """
    Function to send a question to ChatGPT and receive a response.
    :param question: String containing the user's question
    :return: String containing ChatGPT's response or None if an error occurs
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Model can be updated as desired
            messages=[
                {"role": "system", "content": "You are a helpful assistant who answers questions."},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        response_message = response.choices[0].message.content
        return response_message.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Main execution loop
while True:
    with microphone as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            # Recognize speech using Google's speech recognition
            question = recognizer.recognize_google(audio)
            print(f"You asked: {question}")

            # Get and print response from ChatGPT
            response = ask_chatgpt(question)
            if response:
                print(f"ChatGPT: {response}")
            else:
                print("No response received.")

        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
