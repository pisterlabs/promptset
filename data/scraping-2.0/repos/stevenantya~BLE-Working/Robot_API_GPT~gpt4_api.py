from dotenv import load_dotenv
import os
import openai
import json

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "There is 6 location: A,B,C,D,E,Table. I want you to determine what location should I go to. Respond by giving the letter only e.g. A"},
        {"role": "user", "content": "I am having a meeting in location B. Let us go there now!"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "The team is holding a party in location C"}
    ]
)

print(response['choices'][0]['message']['content'])