import time
from openai import OpenAI
from faster_whisper.transcribe import Segment
from typing import Iterable

client = OpenAI()
assistants = client.beta.assistants
chat_completion = client.chat.completions
threads = client.beta.threads
message_thread = client.beta.threads.messages
run_thread = client.beta.threads.runs

# Master Chat Assistant. This is the main chat assistant that will be used to categorize the input and return the data to the correct assistant.
# master_instructions = "You will receive transcribed text from the user. Your task is to clean up any inaccuracies resulting from the speech-to-text process. After cleaning the text, categorize it into one of three types: intent, question, or concept. You are not to engage in conversation or provide answers. Your sole purpose is to identify the type of message and prepare it for the next stage of processing."
# master_return_type = "For each processed message, return the data in the format: 'category: <intent | question | concept>, processed_data: <cleaned text>'."

# master = assistants.create(
#     name="Life Logger Main",
#     instructions=f"{master_instructions} {master_return_type}",
#     tools=[],
#     model="gpt-4-1106-preview",
# )

# # Intent Executor. This is the assistant that will be used to execute the intent of the user.
# intent_instructions = "Upon receiving categorized data marked as 'intent' from the Master Chat Assistant, analyze and execute the user's intended action. Respond with the outcome or the steps taken to execute the intent."

# intent = assistants.create(
#     name="Intent Executor",
#     instructions=intent_instructions,
#     tools=[{"type": "code_interpreter"}],
#     model="gpt-4-1106-preview",
# )

# Concept Logger. This is the assistant that will be used to store all concepts voiced by the user.
# concept_instructions = "Your role is to process and store concepts identified by the Master Chat Assistant. Each concept should be logged, categorized, and stored for future reference. Ensure that the concepts are organized in a manner that facilitates easy retrieval and analysis."

# concept = assistants.create(
#     name="Zettlekasten Obsidian",
#     instructions=concept_instructions,
#     tools=[{"type": "code_interpreter"}],
#     model="gpt-4-1106-preview",
# )

# # Question Answerer. This is the assistant that will be used to answer any questions asked by the user.
# question_instructions = "When you receive a query categorized as a 'question', analyze and provide a comprehensive answer. Your responses should be informative, accurate, and tailored to the user's query."

# question = chats.completions.create(
#     model="gpt-4-1106-preview", stream=True, temperature=0.0
# )
