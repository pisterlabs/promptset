import os
import logging

from openai import OpenAI


logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
