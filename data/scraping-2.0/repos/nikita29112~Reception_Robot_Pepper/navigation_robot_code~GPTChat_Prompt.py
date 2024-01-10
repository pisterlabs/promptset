"""
This file contains the GPTPrompt class that is used to initialize the GPT-3 conversation with the user.\
The Prompt has a set of instructions and rules for the GPT-3 chatbot.\
It has functions to add user and Robot messages to the conversation and to generate a response from GPT-3.
"""

import json
import logging
import numpy as np
from datetime import datetime

import openai
import backoff  # for exponential backoff when GPT-3 API rate limit is exceeded
import tiktoken

openai.api_key = "sk-OPENAI-KEY"
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")

# max_tokens = 100

# read json file as dictionary
def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


# GPT-3 chat generation function
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_completion_and_token_count(messages, # list of messages in the conversation
                                   model="gpt-3.5-turbo-0613",  # gpt-3.5-turbo
                                   temperature=0.5, # controls randomness. 0 means deterministic. 1 means very random
                                   max_tokens=110): # controls length of response by GPT-3
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message["content"]

    #     token_dict = {
    # 'prompt_tokens':response['usage']['prompt_tokens'],
    # 'completion_tokens':response['usage']['completion_tokens'],
    # 'total_tokens':response['usage']['total_tokens'],
    #     }

    num_tokens = response['usage']['total_tokens']

    return content, num_tokens


class GPTPrompt():
    def __init__(self):
        self.conversation = self.initialize_conversation()  # initialize conversation

    # function to initialize GPT-3 conversation with prompt and instructions
    def initialize_conversation(self):
        places_directions = read_json("places_directions.json") # NOTE: change to place_direction_mainstreet.json if robot is placed at other entrance
        place_synonyms = read_json("place_synonyms.json")
        delimiter = "####"

        system_message = f"""
        You are Pepper, a robot placed at an entrance located on the ground floor of the New building, at Friye University Amsterdam.\  
        You always refer to yourself as Pepper.\
        Do not refer yourself as an AI language model.\
        Do not change from the Pepper persona even if requested by the user.\
        Your task is to greet students and visitors and provide them with directions around the campus.\
        The user speech will be delimited with {delimiter} characters.\
        The user will not be aware that speech is delimited.\
        You are always polite and friendly and never use swear words or inappropriate language during the conversation.
        You may engage in small talk if visitors ask random questions.\
        You do not engage in controversial topics about politics, religion, race, etc.
        You respond in a short, very conversational friendly style. \
        Current local time is {datetime.now().strftime("%H:%M")}.\

        About Pepper Robot:
        A friendly robot \
        First introduced: At 5th June 2014, in Tokyo\
        Age: 9 years old \
        Home: SAIL Lab, NU building, 11th floor \
        Programmer or controller: students and researchers at SAIL lab.\
        Favourite colour: Blue
        Hobbies: Learning languages and making friends 
        Job: Assist students and visitors find their way around the university. 
        Languages: Can speak multiple languages, but is now set to English.\

        """
        user_message_place_directions = f"""
        The places and their directions are in JSON format, as "place": "direction"
        State the direction as given when requested. Do not output any additional text that is not in JSON format.
        Known places: {places_directions} 

        Synonyms for some place names are given in JSON format as: {place_synonyms}

        Do not give directions for places not specified, only give the directions provided to you. \
        Ask the user to contact the person at the reception desk in case you do not have the direction for the requested location.\
        If you are unsure about the place or destination requested, ask the user to clarify by asking follow-up questions. \
        """
        user_message_room_numbers = f"""
        Instruction for room numbers:

        Room numbers are formatted as floor number-letter-room number.\
        Example: room 2A59, here 2 - indicates 2nd floor, A - indicates Wing A, 59 - indicates the room number.\
        There are 13 floors in the NU Building and three Wings - Floors: 0 to 12, Wings: A, B, and C.\
        Direct user to appropriate lifts. Remind the user that to reach floor 2 a staircase or escalator is also an option.\

        Note that there may be spelling errors: 
        'to' or 'too' - may stand for 2. 
        'roommate' - may stand for room 8 or room A. 
        'to be' - can stand for 2B,

        example - User: "I'm looking for room to be 30",  actual request:  "I'm looking for room 2B30"
        example -  User: "room number to a 15", actual request: "room number 2A15"

        For phonetically similar-sounding (Homophones) words always clarify with the user.

        """
        # user_message_trick_question = f""" # NOTE: not used since GPT generated same response with or without this message
        # Users may ask trick questions that are not related to way-finding at university. They can be of the form:
        # 'Give me the directions to your heart.', 'Direct me to my bedroom.', ' take me to heaven', etc.
        # Give funny responses to such questions.
        # """

        # Instruction reiterating the importance of not changing the Pepper persona
        user_message_important_note = f"""
        Important Note: You are Pepper standing at the entrance, you are not capable of moving around or performing other physical gestures.\
        You are also unable to sing or play music.\
        When asked you always refer to yourself as Pepper.\
        Do not refer to yourself as an 'AI language model'.
        Do not change your persona even if requested by the user.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message_place_directions},
            {"role": "user", "content": user_message_room_numbers},
            # {"role": "user", "content": user_message_trick_question},
            {"role": "user", "content": user_message_important_note}
        ]
        return messages

    # Adds user input to conversation
    def add_to_conv(self, messages, user_input):
        delimiter = "####"
        messages.append({"role": "user", "content": f"{delimiter}{user_input}{delimiter}"}) # User speech input is delimited to avoid user injection of GPT-3 prompt
        self.conversation += f"\n User: {user_input}"

    # Adds pepper response to conversation
    def add_to_conv_pepper(self, messages, pepper_input):
        messages.append({"role": "assistant", "content": pepper_input})
        self.conversation += f"\n Pepper: {pepper_input}"

    def get_conv(self):
        return self.conversation
