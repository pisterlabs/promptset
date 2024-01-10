from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat(message, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message["content"]


# Text insertion
result = chat("The [blank] refers to the rule by the father.")
print(result)

prompt = """I kayaked across the Hudson.

Please identify the name of the river in this passage.
"""

response = chat(prompt)
print(response)

prompt = """Please identify the name of the river in the following passages:
1. I kayaked across the hudson
2. I swam across the Mississippi
"""
response = chat(prompt)
print(response)
