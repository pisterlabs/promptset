import os
from dotenv import load_dotenv
from langchain.llms import OpenAI


load_dotenv('./.env')

active_looptoken = os.getenv('ACTIVE_LOOPTOKEN')
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(model="text-davinci-003", temperature=0.3)

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
print(llm(text))
