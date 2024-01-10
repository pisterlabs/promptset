import os
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# models = [
#     "gpt-3.5-turbo",
#     "gpt-4",
#     "gpt-4-1106-preview",
# ]

# Set a default model
current_model = "gpt-4-1106-preview"

# Initialize chat history
messages = []

while True:
    # Display chat messages from history
    for message in messages:
        print(f"{message['role'].title()}: {message['content']}")

    # React to user input
    prompt = input("What is up? ")

    if prompt:
        # Add user message to chat history
        messages.append({"role": "user", "content": prompt})

        # Create stream by sending user message to OpenAI API
        completion_stream = client.chat.completions.create(
            model=current_model,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            stream=False,  # Set to False to wait for full response
        )

        # Process the full response from OpenAI API
        full_response = completion_stream["choices"].message['content']

        # Display OpenAI API response
        print(f"Assistant: {full_response}")

        # Add assistant message to chat history
        messages.append({"role": "assistant", "content": full_response})
