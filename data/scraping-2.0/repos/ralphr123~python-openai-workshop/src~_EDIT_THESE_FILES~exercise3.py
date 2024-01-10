# Load environment variables from .env
from dotenv import load_dotenv; load_dotenv()

# Typings are good practice to know what your code is producing
from typing import List

# OpenAI Python SDK to allow us to make requests to GPT
import openai

# ===========================
# === MODIFY THIS SECTION ===
# ===========================

# List to store message history
history = []

# Add a message of role "user" to the history array
def addUserMessage(text: str):
	history.append({ 'role': 'user', 'content': text })

# Add a message of role "system" to the history array
def addSystemMessage(text: str):
	raise NotImplementedError('addSystemMessage() has not been implemented yet') # Remove this line when you start working on this function

# Add a message of role "assistant" to the history array
# You can copy this function from exercise2.py
def addAIMessage(text: str):
	raise NotImplementedError('addAIMessage() has not been implemented yet') # Remove this line when you start working on this function

# Send a message to GPT and return the response string
# You can copy this function from exercise2.py
def sendMessage(text: str) -> str:
	raise NotImplementedError('sendMessage() has not been implemented yet') # Remove this line when you start working on this function

# Get all non-system messages from array. This should return a List of messages.
def getChatMessages() -> List:
	# Feel free you use any method you'd like to filter out the system message
	# Ex. loop, python filter function, subarrays, etc.
	raise NotImplementedError('getChatMessages() has not been implemented yet') # Remove this line when you start working on this function
	