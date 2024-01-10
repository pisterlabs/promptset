import os
from pathlib import Path
import file_operations as fo
from dotenv import load_dotenv
from openai import OpenAI

# Get environment variables
path = Path("Environment-Variables/.env")
load_dotenv(dotenv_path=path)

# Setup OpenAI client
client = OpenAI(
    organization=os.getenv('organization'),
    api_key=os.getenv("api_key")
)

# Get category
category = input("What category would you like to generate a story from? ")

# Get a random prompt from the category
prompt = fo.read_category(category)
prompt = "Use the following prompt to generate an interactive story. Ask a question at the end of each response, and let the user respond with what they would do: " + prompt

conversation_history = [{"role": "user", "content": prompt}]

# Set up the starting GPT prompt
response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=conversation_history
)

# Print the response
print(response.choices[0].message.content)
conversation_history.append({"role": "system", "content": response.choices[0].message.content})

# Start the user/GPT interaction
while True:
    # Get the user's response
    user_response = input("What would you do? ")

    conversation_history.append({"role": "user", "content": user_response})
    print(conversation_history)

    # Generate a response from GPT
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=conversation_history
    )

    # Print the response
    print(response.choices[0].message.content)

    conversation_history.append({"role": "system", "content": response.choices[0].message.content})
