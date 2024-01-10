import environ
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory

env = environ.Env()
environ.Env.read_env()
API_KEY = env("OPENAI_API_KEY")

chat = ChatOpenAI(temperature=0, openai_api_key=API_KEY)

history = ChatMessageHistory()

history.add_ai_message("Hello, how are you?")

history.add_user_message("What is the capital of Russia?")


print(history.messages)

ai_response = chat(history.messages)

print(ai_response)
