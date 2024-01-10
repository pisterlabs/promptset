# This is the main file for the Aisha speech interface
# It uses the OpenAI API to generate responses to user input:
#   You will need to set up an account and get an API key from https://beta.openai.com/
#   In the root directory of this project, create a file called .env
#   In the .env file, add the following line:   OPENAI_API_KEY="your-api-key"

#Speech recognition commands
# adjust sensitivity (to adjust the sensitivity of the microphone)
# exit (to exit the program)
from config.basic_config import ai_personality as personality
import os
from gtts import gTTS
import playsound
from dotenv import load_dotenv
import openai
from functions.speech_functions import adjust_sensitivity, speech_to_text

load_dotenv() # take environment variables from .env.
token = os.getenv("OPENAI_API_KEY") # Accessing variables.

openai.api_key = token  # Set the API key directly in the openai module

while True:

    # Get user input
    #say_to_aisha = input("Say something to Aisha (type 'exit' to end): ")
    say_to_aisha = speech_to_text()

    # Check if the user wants to exit
    if say_to_aisha.lower() == 'exit':
        break  # exit the loop

    if say_to_aisha == "adjust sensitivity":
        adjust_sensitivity()
        continue

    if say_to_aisha == "timeout":
        message_content = "I'm lonely, are you ignoring me"

    elif say_to_aisha != "unrecognised":
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": personality},
                {"role": "user", "content": say_to_aisha},
            ]
        )
        message_content = response.choices[0].message.content
    else:
        continue

    # print(message_content)
    sound=gTTS(text=message_content, lang='en', slow=False) # text to speech(voice)
    sound.save("sound.mp3") # save the audio file as sound.mp3
    playsound.playsound("sound.mp3", True) # True to play asynchronously
    os.remove("sound.mp3")