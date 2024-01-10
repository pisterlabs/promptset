from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

print(OpenAI().predict("Hello how are you?").strip())
