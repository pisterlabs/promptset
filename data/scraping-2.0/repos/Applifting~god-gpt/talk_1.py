from dotenv import load_dotenv
load_dotenv()
import os
from langchain.llms  import OpenAI
from voice.speech import speak

openai_api_key = os.getenv("OPENAI_API_KEY")

def talk_to_ai():
    llm = OpenAI(openai_api_key=openai_api_key, verbose=True)
    text = llm("Hey, are you there?")
    speak(text, "Bella")



talk_to_ai()