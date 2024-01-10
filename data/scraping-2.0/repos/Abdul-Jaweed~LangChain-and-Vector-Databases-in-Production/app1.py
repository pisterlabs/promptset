from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=api_key, 
             model="text-davinci-003", 
             temperature=0.9
             )

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."

print(llm(text))