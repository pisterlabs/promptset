import os
import openai
import dotenv
import prompts

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# print out all available engines
engines = openai.Engine.list()
print(engines)