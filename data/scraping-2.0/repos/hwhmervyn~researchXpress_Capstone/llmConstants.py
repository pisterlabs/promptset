import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# Read local ".env" file stored in top-most directory to retrieve API key
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

# Instantiate LLM

# GPT-3.5-turbo model
chat = ChatOpenAI(temperature=0, # lower temperature --> more predictable responses
            model_name = 'gpt-3.5-turbo')

# text-davinci-003 model (to be deprecated in Jan 2024)
llm = OpenAI(temperature=0)