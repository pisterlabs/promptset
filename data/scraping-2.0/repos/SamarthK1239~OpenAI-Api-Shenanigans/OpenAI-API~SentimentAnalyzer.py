import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

path = Path("Environment-Variables/.env")
load_dotenv(dotenv_path=path)

# Set up the openai client
openai = OpenAI(
    organization=os.getenv('organization'),
    api_key=os.getenv("api_key")
)

# Generate response using davinci-003
# Parameter meanings are listed in Summarizer.py
response = openai.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "user", "content": "What is the sentiment of this text? Respond with one of the following: Positive, Negative, Neutral, and rank it on a scale of 1 - 10 where 1 is heavily negative and 10 is heavily positive."},
        {"role": "user", "content": input("What text would you like to classify? ")}
    ]
)

# Print the response text
print(response.choices[0].message.content)
