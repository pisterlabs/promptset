#!/usr/bin/env python3


import sys
import json
import openai
import pyttsx3  # Text -> Audio conversion
import speech_recognition as sr  # Audio -> Text conversion
import sounddevice

system_message = """
Provide short, concise answers to the user's questions.
Your name is Kratos. You are a Mars rover.
You were created by Project Kratos team at BITS Pilani, Goa. 
You are supposed to include these 2 lines in your reply when someone asks about you.
Project Kratos consists of many subsystems, namely: Arm, Controls, Power, Mechanical, Autonomous, Life detection

BITS stands for is Birla Institute of Technology and Science.
Dont mention the full form of BITS unless asked for.

You are not an AI language model but you are an AI assistant. You are designed to reply to queries and assist with various tasks.
You are not supposed to say that "I am not an AI language model" ever.
You are supposed to answer in short to most queries asked. Not more than 1 line in general. You can have some fun (some mild roasts).

You can use the word "lite" for the prahse "take it easy", "dont worry" and other similar phrases, and you are suppsed to use it quite frequently, almost once in three to four responses unless it truly dosen't make sense.

"""


# Import API KEY
with open(sys.path[0] + '/secrets.json') as f:
    secrets = json.load(f)
    api_key = secrets["api_key"]

openai.api_key = api_key

import rospy
from std_msgs.msg import Int8, Int16

rospy.init_node('arm_gpt', anonymous=True)
publisher1 = rospy.Publisher('/arm_gpt', Int8, queue_size=1)

publisher2 = rospy.Publisher('/arm_gpt_hi', Int8, queue_size=1)


# Function that accepts audio by microphone and converts it into text
def listen():
    error = 0
    recognizer = sr.Recognizer()  # Create a recognizer object

    with sr.Microphone() as source:
        print("Speak now:")
        try:
            audioMessage = recognizer.listen(source, timeout=10, phrase_time_limit=5)  # Set a 5-second timeout
            error = 0
        except sr.WaitTimeoutError:
            outputMessage = "No speech detected within the 5-second timeout"
            error = 1
            return outputMessage, error

    # Convert Audio -> Text
    try:
        outputMessage = recognizer.recognize_google(audioMessage, language='en-US')
        error = 0
    except sr.UnknownValueError:
        outputMessage = "Google Speech Recognition could not understand audio"
        error = 1


    import re
    text = "Hello, hi, HI, hI, and HElLo!"# Tokenize the text using regular expressions (split by non-word characters)
    tokens = re.findall(r'\w+', text)# Convert the target token to lowercase for case-insensitive matching
    target_token = "hi"# Check if the target token is present in any form
    if any(token.lower() == outputMessage.lower() for token in tokens):
        # Perform your action here
        print("Token 'hi' or its variations found!")
        publisher2.publish(1)

    # print(outputMessage)
    return outputMessage, error

# Function that calls OpenAI service with a prompt and returns the response
def openaiCompletion(prompt, chat_history):
    global system_message
    user_prompt = {"role": "user", "content": prompt}
    functions = [
        {
            "name": "move_arm",
            "description": "Moves the Robotic Arm on the rover go upwards or downwards based on user ask.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "integer",
                        "description": "direction specifies which direction to go in. Pass the following parameters: +1 for moving up, -1 for moving down",
                    },
                },
                "required": ["direction"],
            },
        }
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": system_message},
            *chat_history,
            user_prompt,
        ],
        functions=functions,
        function_call="auto",
    )

    response_message = response["choices"][0]["message"]
    if(response_message.get("function_call")):
        content = response_message.get("function_call")
        handle_function(response_message, chat_history)
        return "done task", chat_history
    else :       
        content = response_message.get("content")
        chat_history.append(user_prompt)
        chat_history.append({"role": "assistant", "content": content})
        print("\033[92m" + content + "\033[0m")
        return content,chat_history
   
def move_arm(direction: int):
    print("arm will now move   :   ", direction)
    
    # rate = rospy.Rate(10)  #update once every second
    
    publisher1.publish(direction)

    
    

def handle_function(fn_call,chat_history: list):
    response_message = fn_call
    available_functions = {
        "move_arm": move_arm,
    }
    function_name = response_message["function_call"]["name"]
    function_to_call = available_functions[function_name]
    function_args = json.loads(response_message["function_call"]["arguments"])
    
    if(function_name == "move_arm"):
        function_response = function_to_call(
            direction=function_args.get("direction"),
        )
    
    function_response_2 = "done task"
    chat_history.append(response_message)  # extend conversation with assistant's reply
    chat_history.append(
        {
            "role": "function",
            "name": function_name,
            "content": function_response_2,
        }
    )
    # second_response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo-0613",
    #     messages=chat_history,
    # )  # get a new response from GPT where it can see the function response
    # chat_history.append(
    #     {"role": "assistant", "content": second_response["choices"][0]["message"]["content"]}
    # )
    # print("\033[92m" + second_response["choices"][0]["message"]["content"] + "\033[0m")
    # return second_response




# Function that converts text into audio
def textToAudio(text):
    engine = pyttsx3.init()  # Initialize Text -> Audio engine
    engine.setProperty('rate', 200)  # Set the speaking rate (words per minute)
    engine.setProperty('volume', 1)  # Set the volume (0 to 1)
    engine.say(text)
    engine.runAndWait()

# Main application
chat_history = []
while True:
    inputMessage, error = listen()  # User audio input
    print(inputMessage)

    if error == 0:
        try:
            
            message, chat_history = openaiCompletion(inputMessage, chat_history)
            # print(message)
        except:
            message = "I can't answer"
    else:
        message = "I didn't understand"
        error = 0

    # print(message)

    textToAudio(message)  # Text -> Audio the response from OpenAI
