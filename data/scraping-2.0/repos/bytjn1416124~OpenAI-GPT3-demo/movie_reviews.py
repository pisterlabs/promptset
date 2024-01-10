# -*- coding: utf-8 -*-
import openai
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEYY')
from generalGpt3 import GPT3

prompt_en =   """This is how a movie review would sound like if it was written by Jane Austen.

Movie: """
prompt_no =   """"""

GPT3(prompt_en, "Movie:", "Review", temperature=0.7,frequency_penalty=0.8,presence_penalty=0.6)
