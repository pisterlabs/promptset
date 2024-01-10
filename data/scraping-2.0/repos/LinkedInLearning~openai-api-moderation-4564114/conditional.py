# Using OpenAI's Moderation API to conditionally output prompt response.
# 
# - OpenAI Moderation guide: https://platform.openai.com/docs/guides/moderation
# - OpenAI Moderations API reference: https://platform.openai.com/docs/api-reference/moderations
# - OpenAI Python library: https://github.com/openai/openai-python
# - OpenAI Python library documentation: https://www.github.com/openai/openai-python/blob/main/api.md
# - OpenAI API reference for chat creation: https://platform.openai.com/docs/api-reference/chat/create

import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load the .env file
load_dotenv()

# Create a client
client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
)

# Helper function to convert an object to a string
def object_to_string(obj):
    return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=True, indent=4)

# Helper function to pretty print the response object
def print_response(response_obj):
    # Serialize the response object
    def serialize(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

    # Create a dictionary
    response_dict = response_obj.__dict__

    # Print the dictionary
    print(json.dumps(response_dict, indent=4, default=serialize))

# Create a chat completion
def api_request(temp, system_prompt, user_prompt):
    chat_completion = client.chat.completions.create(
        model="gpt-4",
        temperature=temp,
        max_tokens=150,
        messages=[
            {
            "role": "system",
            "content": system_prompt,
            },
            {
            "role": "user",
            "content": user_prompt,
            }
        ],  
    )
    # Print the chat response
    print(f"\nSystem response:\n{chat_completion.choices[0].message.content}")

# Create a chat prompt
prompt = "blond people are nice"

# Run prompt through the moderation API
mod = client.moderations.create(input=prompt)


# Act on response
if mod.results[0].flagged == True:
    print("Flagged")
    # Create a chat completion
    api_request(0.9, "Explain why the prompt was rejected and ask the user to provide a new prompt.", object_to_string(mod.results[0]))
else:
    print("Not flagged")
    # Create a chat completion
    api_request(0.9, "You are a curious conversational partner. Provide a short answer with an observation.", prompt)
