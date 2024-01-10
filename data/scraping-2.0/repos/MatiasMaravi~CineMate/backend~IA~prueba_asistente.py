import openai
import os

client = openai
openai.api_key = os.getenv("API_KEY")


my_assistant = openai.ChatCompletion.create(
    instructions="You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
    name="Math Tutor",
    tools=[{"type": "code_interpreter"}],
    model="gpt-3.5-turbo-1106",
)
print(my_assistant)
