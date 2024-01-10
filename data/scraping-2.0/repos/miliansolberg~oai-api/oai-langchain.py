from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="vars/.env")

api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(openai_api_key=api_key)

result = chat_model.predict("How is quantum computing going to advance artificial intelligence?")
print(result)