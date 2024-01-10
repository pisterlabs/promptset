from open_ai_handler import OpenAIHandler as handler
from dotenv import load_dotenv
import os

load_dotenv()

api = os.getenv("OPENAI_KEY")
ai = handler(key=api)

while True:
    prompt = input("You: ")
    print("Bot: " + ai.message(prompt))
    