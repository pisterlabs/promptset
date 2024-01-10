"""
GPT voice assistant

1. user says "Hey GPT"
2. then says a command, command is transcribed to text
3. GPT responds to the command
4. use tts to play the response
"""

import os
import json
import openai
import pyttsx3 # uses system TTS so its fast but not very natural, alternatives are: gTTS, CoquiTTS, larynx, Bark
import dotenv
import asyncio
import speech_recognition as sr
from functions import functions, available_functions

# load environment variables
dotenv.load_dotenv()

# set openai api key
openai.api_key = os.getenv("OPENAI_API_KEY")

# keep track of conversation
conversation = []

# setup conversation mode
conversation_init = {"role": "system", "content": "You are a helpful voice assistant. You will respond to the user's requests."}

# set up conversation
conversation.append(conversation_init)

# initialize pyttsx3 audio engine
AUDIO_ENGINE = pyttsx3.init()
AUDIO_ENGINE.setProperty('rate', 125) # set speech rate
AUDIO_ENGINE.setProperty('volume', 1.0) # set speech volume

# get voices
voices = AUDIO_ENGINE.getProperty('voices')

# set voice
AUDIO_ENGINE.setProperty('voice', voices[1].id) # female voice, voices[0].id is male

# get gpt model from environment variable
MODEL = os.getenv("GPT_MODEL")

# create recognizer instance
recongizer = sr.Recognizer() 

def detect_microphone():
    """
    Detect microphone
    """

    print("Say something to detect microphone ...")

    for device_index in sr.Microphone.list_working_microphones():
        m = sr.Microphone(device_index=device_index)
        print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(device_index, m))
        break
    else:
        print("No working microphones found!")
        exit(1)

def speech_to_text():
    """
    Transcribes speech to text
    """

    with sr.Microphone() as source: # use the default microphone as the audio source
        source.pause_threshold = 1 # seconds of non-speaking audio before a phrase is considered complete
        
        recongizer.adjust_for_ambient_noise(source) # adjust for ambient noise
        
        print("Listening...")
        
        audio = recongizer.listen(source) # record audio prompt

        try:
            # convert speech to text
            text = recongizer.recognize_google(audio, show_all=True)
            if text:
                text = text['alternative'][0]['transcript'] # get first transcript
                return text

        except sr.UnknownValueError:
            print('No speech detected')

        except sr.RequestError:
            print('API was unreachable or unresponsive')

        return ""

def text_to_speech(text):
    """
    Converts text to speech
    """

    AUDIO_ENGINE.say(text)
    AUDIO_ENGINE.runAndWait()

def delete_conversation():
    """
    Delete conversation
    """

    global conversation
    global conversation_init
    conversation = []
    conversation.append(conversation_init)

def chatgpt_response():
    """
    Get response from chatgpt
    """

    try:

        # get chat response
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=conversation,
            functions=functions,
            function_call='auto'
        )

        print('ChatGPT response: ', response['choices'][0]['message'])

        response_message = response["choices"][0]["message"]

        # check if function call
        if response_message.get("function_call"):

            function_name = response_message["function_call"]["name"]
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])

            function_response = fuction_to_call(
                **function_args
            )

            # add assistant response to call the function
            conversation.append(response_message)

            # extend conversation with function response
            conversation.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )

            # get second response with function response
            second_response = openai.ChatCompletion.create(
                model=MODEL,
                messages=conversation,
            )

            # assistant response to function response
            conversation.append({"role": "assistant", "content": second_response['choices'][0]['message']})

            return second_response['choices'][0]['message']

        else:
            # add assistant response to conversation
            conversation.append({"role": "assistant", "content": response['choices'][0]['message']})

            return response['choices'][0]['message']

    except Exception as e:
        print('Error: ', e)
        return False

async def main():
    # uncomment to debug microphone
    #detect_microphone()

    print("Say 'Hey GPT' to start the conversation ...")

    while True: # run in loop
        text = speech_to_text()

        print("Text: " + text)

        if text.lower == "delete conversation":
            delete_conversation()
            text_to_speech("Conversation deleted.")

        if text.lower() == "hey gpt": # use lower case to avoid case sensitivity
            print("Listening for command ...")

            text = speech_to_text() # get command

            if text:
                print("Human: " + text)

                # add user prompt to conversation
                conversation.append({"role": "user", "content": text})

                response = chatgpt_response()

                if response:
                    text_to_speech(response)
                else:
                    text_to_speech("Sorry, I did not get that.")

if __name__ == "__main__":
    asyncio.run(main())