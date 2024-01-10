# Load environment variables from .env
from dotenv import load_dotenv; load_dotenv()

# Typings are good practice to know what your code is producing
from typing import List

# OpenAI Python SDK to allow us to make requests to GPT
import openai

# Use this function to print to the console for debugging
def printToConsole(*args, sep=' ', end='\n'):
    print(*args, sep=sep, end=end, flush=True)

# ===========================
# === MODIFY THIS SECTION ===
# ===========================

# List to store message history
history = []

# Add a message of role "system" to the history array
def addSystemMessage(message: str) -> None:
	history.append({'role': 'system', 'content': message})

# Add a message of role "user" to the history array
def addUserMessage(message: str) -> None:
	history.append({'role': 'user', 'content': message})

# Add a message of role "assistant" to the history array
def addAIMessage(message: str) -> None:
	history.append({'role': 'assistant', 'content': message})

# Send a message to GPT and return the response string
def sendMessage(message: str) -> str:
	# 1. Use the function you completed to add the message as a user message
	addUserMessage(message)

	# 2. Call Azure to send the message to GPT
	res = openai.ChatCompletion.create(
		engine='gpt-35-turbo',
		messages=history,
	)

	# 3. Store the result in a variable
	message = res['choices'][0]['message']['content']
	
	# 4. Use the function you completed to add the response as an AI message
	addAIMessage(message)

	# 5. Return the response string
	return message

# Get all non-system messages from array
def getChatMessages() -> List:
	# Feel free you use any method you'd like to filter out the system message
	# Ex. loop, python filter function, subarrays, etc.
	
	chatMessages = []

	for message in history:
		if message['role'] != 'system':
			chatMessages.append(message)
	
	return chatMessages