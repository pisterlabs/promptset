# -*- coding: utf-8 -*-
"""
Project: Quick-and-Dirty GPT-3 Q&A Bot
Author: Ted Hallum
Date: 28 NOV 2021
"""
# =============================================================================
# IMPORTS
# =============================================================================
from dotenv import load_dotenv
import openai
import os

# =============================================================================
# USER DEFINED FUNCTIONS
# =============================================================================
# Sends requests to OPENAI's API
def ask():
    prompt_text = f'{dialogue}{question_preface}: {question}{answer_preface}:'
    response = openai.Completion.create(
        # The ID of the engine to use for this request
        # Use openai.Engine.list() to get the list of available engine IDs.
        engine ="davinci",
        # ---------------------------------------------------------------------
        # The prompt(s) to generate completions for
        prompt = prompt_text,
        # ---------------------------------------------------------------------
        # Number between 0 and 1.
        # Higher values means the model will take more risks.
        temperature = .05,
        # ---------------------------------------------------------------------
        # Max number of tokens to generate in the completion.
        # Rule of thumb: One token generally corresponds to ~4 characters of 
        # common English text. So, 100 tokens ~= 75 words.
        max_tokens = 150, 
        # ---------------------------------------------------------------------
        # Number between 0 and 1.
        # An alternative to temperature. Use temperature or top_p, but not both.
        # top_p = 1,
        # ---------------------------------------------------------------------
        # How many completions to generate for each prompt.
        # NOTE: This parameter can quickly consume your token quota. Use 
        # carefully in combination with prudent settings for max_tokens and stop.
        n = 1,
        # ---------------------------------------------------------------------
        # Number between -2.0 and 2.0. Positive values increase the model's 
        # likelihood to talk about new topics.
        presence_penalty = 0,
        # ---------------------------------------------------------------------
        # Number between -2.0 and 2.0. Positive values decrease the model's 
        # likelihood to repeat the same line verbatim.
        frequency_penalty = 0,
        # ---------------------------------------------------------------------
        # Up to 4 sequences where the API will stop generating further tokens. 
        # The returned text will not contain the stop sequence.
        stop = ['\n']
        )
    
    answer = response['choices'][0]['text']
    
    return str(answer)

# Give the Q&A bot memory
def append_interaction_to_dialogue():
    return f'{dialogue}{question_preface} {question}{answer_preface}{answer}'

# =============================================================================
# SET OPENAI API KEY
# =============================================================================
# Change current working directory to script location
os.chdir(os.path.abspath(os.path.dirname(__file__)))

# Load environment variables from .env file
load_dotenv()

# Set API key
openai.api_key = os.environ['OPENAI_API_KEY']

# =============================================================================
# SET CONVERSATION PARAMETERS
# =============================================================================
# Global variables used to teach the bot how to speak and answer questions
answer_preface = '\nA:'
question_preface = '\n\nQ:'
initial_prompt = 'I am a highly intelligent question answering bot. If you '\
                 'ask me a question that is rooted in truth, I will give you '\
                 'a very detailed answer. If you ask me a question that is nonsense, '\
                 'trickery, or has no clear answer, I will respond with '\
                 '\"Unknown\".\n\nQ: What is human life expectancy in the '\
                 'United States?\nA: Human life expectancy in the United '\
                 'States is 78.5 years.\n\nQ: Who was president of the United '\
                 'States in 1955?\nA: Dwight D. Eisenhower was president of '\
                 'the United States in 1955.\n\nQ: Which party did he belong '\
                 'to?\nA: He belonged to the Republican Party.\n\nQ: What is '\
                 'the square root of banana?\nA: Unknown\n\nQ: How does a '\
                 'telescope work?\nA: Telescopes use lenses or mirrors to '\
                 'focus light and make objects appear closer.\n\nQ: Where '\
                 'were the 1992 Olympics held?\nA: The 1992 Olympics were '\
                 'held in Barcelona, Spain.\n\nQ: How many squigs are in a '\
                 'bonk?\nA: Unknown'
dialogue = None
question = None

# =============================================================================
# BOT INTERACTION
# =============================================================================
# If you want GPT-3 to be contextually aware of past answers and 
# "keep its story straight", then set mem_mode = 'y'
mem_mode = input("Enable memory mode ('y' or 'n')? ")

# Q&A loop
while question != 'exit':
    if dialogue == None:
        dialogue = initial_prompt
        print('\n### WELCOME TO THE QUICK-AND-DIRTY GPT-3 Q&A BOT! ###')
        print("\nSubmit your question (when done inquiring, type 'exit').")
    question = input ("QUESTION: ")
    if question != 'exit':
        answer = ask()
        print(f'ANSWER:{answer}')
        if mem_mode == 'y':
            dialogue = append_interaction_to_dialogue()
    else:
        print('\nThank you for your questions! Goodbye!')
