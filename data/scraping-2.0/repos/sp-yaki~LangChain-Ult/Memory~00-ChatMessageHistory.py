from langchain.memory import ChatMessageHistory

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

history = ChatMessageHistory()

history.add_user_message("Hello, nice to meet you.")

history.add_ai_message("Nice to meet you too!")

print(history)