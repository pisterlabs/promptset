"""
Darin Kishore, dakisho
This code was my own Work. It was written without consulting
sources outside of those provided by the instructor.
"""


from emora_stdm import DialogueFlow
import macros
import spacy
import time
import requests
import json
import os
import sqlite3
import openai
from re import Pattern
from utils_babel import MacroGPTJSON, MacroNLG, MacroGPTJSONNLG, gpt_completion
from macros_babel import macros

import re
from emora_stdm import Macro, Ngrams
from typing import Dict, Any, List
from enum import Enum

PATH_API_KEY = '../resources/openai_api.txt'
def api_key(filepath=PATH_API_KEY) -> str:
    fin = open(filepath)
    return fin.readline().strip()


openai.api_key = api_key()


# code executes left to right

# what was your favorite part, what was your favorite part, why do you like that character, did you like the movie

introduction = {
    'state': 'start',
    '`Hi! What\'s your name?`': {
        '#SET_CALL_NAME': {
            '`It\'s nice to meet you,`#GET_CALL_NAME`! My favorite movie is Babel! Do you like Babel?`': {
                'error': 'babel_next'
            }
        }
    }
}


# pretreatment, early in treatment, late in treatment

babel_next= {
    'state': 'babel_next',
    '#BABEL_RESPONSE': {
        'error': 'babel_next'
    }
}

babel_test = {
    'state': 'babel_questions',
    '`It is my favorite of all time! What is your favorite part of the movie?`': {
        'error': 'babel_next'
    }
}


df = DialogueFlow("start", end_state="end")
df.load_transitions(introduction)
df.load_transitions(babel_test)
df.load_transitions(babel_next)
df.add_macros(macros)

if __name__ == "__main__":
    df.run()
