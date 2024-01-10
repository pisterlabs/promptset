import speech_recognition as sr
import pyttsx3
import openai
import json
from gtts import gTTS
import os
from pygame import mixer

# Initialize recognizer and the text to speech engine
r = sr.Recognizer()
engine = pyttsx3.init()

# Get the list of voices
voices = engine.getProperty('voices')

# Set the voice you want to use by its ID
desired_voice_id = voices[2].id  # Replace 0 with the index of the voice you want to use
engine.setProperty('voice', desired_voice_id)

# Set your OpenAI API key here
openai.api_key = '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'

# Load the conversation history from a file
try:
    with open('history.json', 'r') as f:
        messages = json.load(f)
except FileNotFoundError:
    # If the file doesn't exist, initialize the list of messages with the system message
    messages = [{"role": "system", "content": "You're name's Sofiya.You are a kinda bot, created by MK.You have the ability to remember personal details.You are created in Bagya Matriculation School.Your Developer Name is MK.He is an indie-Developer Who creates amazing games, applications, Animations, Websites, and some programmings.You're Created in Sri Bagya Matriculation School by MK.You must thankful for the teachers who supported in creating you.You're owner's name is MK.As an AI you have a name but not age or anything else.You are created esspeciealy for a Science Exhibition as his project.If you cant answer the question you just simply say, Ask MK.As well as you can speak throuh spoken words you only support english"}]

# Initialize a global counter for the audio files
audio_file_counter = 0

# Function to convert text to speech in Tamil using gTTS and play it using pygame
def speak(text):
    global audio_file_counter  # Use the global counter variable
    print(text)  # Print the text to the console
    tts = gTTS(text=text, lang='ta')
    directory = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
    result_audio_directory = os.path.join(directory, 'result_audio')  # Join the directory with the folder name
    if not os.path.exists(result_audio_directory):  # Check if the directory exists
        os.makedirs(result_audio_directory)  # If not, create it
    filename = os.path.join(result_audio_directory, f'tamil_audio_{audio_file_counter}.mp3')  # Join the directory with the filename and counter
    tts.save(filename)
    mixer.init()
    mixer.music.load(filename)
    mixer.music.play()
    while mixer.music.get_busy():  # wait for the audio to finish playing
        pass

    audio_file_counter += 1  # Increment the counter

# Function to listen to the microphone and recognize speech in Tamil
def listen():
    with sr.Microphone() as source:
        speak("கேட்கிறேன்...")  # "Listening..." in Tamil
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language='ta-IN')
            print(f"You Said : {text}")
            return text
        except:
            speak("மன்னிக்கவும், நான் அதை பெறுவதில் ஏதோ சிக்கல் உள்ளது.")  # "Sorry, I didn't get that" in Tamil
            return listen()

# Main function to process commands and interact with GPT-3 model based on command.
def main():
    while True:
        command = listen()

        # Add the user's question to the list of messages.
        messages.append({"role": "user", "content": command})

        # Get the answer from ChatGPT.
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages  # Use the list of messages here.
        )

        # Speak out and print the answer.
        answer = response['choices'][0]['message']['content']
        speak(answer)

        # Add the assistant's answer to the list of messages.
        messages.append({"role": "assistant", "content": answer})

        # Saving the conversation history to a file.
        with open('history.json', 'w') as f:
            json.dump(messages, f)

if __name__ == "__main__":
    main()
