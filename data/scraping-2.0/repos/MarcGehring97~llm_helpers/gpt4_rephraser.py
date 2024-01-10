import sys
import json
import os
import openai
from openai import OpenAI
# Idea from: https://www.youtube.com/watch?v=CluNm3OfyO8&ab_channel=IamYou
# Find the quick actions at ~/Library/Services

# Load OpenAI API key from file
os.environ["OPENAI_API_KEY"] = "<your OpenAI API key>"

# Load OpenAI API key from environment variable for security reasons
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("No OpenAI API key provided.")

# Instantiate OpenAI client
client = OpenAI()

# Get the command line arguments
args = sys.argv[1:]

# Convert the arguments to a single string
user_content = ' '.join(args)

def get_reponse(prompt):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        print(chunk.choices[0].delta.content or "", flush=True, end="")

prompt = f"""
    Rephrase the following sentences to be more reader-friendly and engaging in the same language as the user content. Do not preface your response with Response, provide the improved sentence directly.

    Examples:
    User content: "I have a test tomorrow, so I needs to study all night. I'm so tired."
    Response: "I have a test tomorrow, so I need to study all night. I'm so tired."

    User content: "He don't like to eat vegetables. Vegetables are healthy and provide essential nutrients."
    Response: "He doesn't like to eat vegetables. Vegetables are healthy and provide essential nutrients."

    User content:
    {user_content}
"""

# Extract the 'content' field from the response
get_reponse(prompt)
