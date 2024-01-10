# -*- coding: utf-8 -*-
import openai
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEYY')
from generalGpt3 import GPT3


prompt_en =   """Imdb is a large movie database where the users can rate the movies from 1 to 10.
Here are some examples of ratings at imdb:

Movie: Jurrasic Park
Rating: 8.1

Movie: Inception
Rating: 7.8

Movie: The good, the bad and the ugly
Rating: 8.8

Movie: Charlie and the chocolate factory
Rating: 6.6

Movie: Tenet
Rating: 7.4

Movie: The room
Rating: 3.7

Movie: James Bond: Spectre
Rating: 6.8

Movie: 10 things I hate about you
Rating: 7.3

Movie:"""

prompt_no =   """"""
GPT3(prompt_en, "Movie:", "Rating:",temperature=0.7, frequency_penalty=0.8, presence_penalty=0.6)
