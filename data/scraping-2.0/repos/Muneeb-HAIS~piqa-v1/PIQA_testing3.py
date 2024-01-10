import os
from openai import OpenAI
import pandas as pd

os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()

# Your code using the OpenAI client
df = pd.read_excel(r'C:\Users\acer1\Downloads\Company Structure Dataset.xlsx')

# Convert DataFrame to JSON string
df_json = df.to_json(orient='split')

# System message to set the behavior of the assistant
system_message = {"role": "system", "content": df_json}

# User messages for interaction
user_messages = [
    {"role": "user", "content": "Hello, who should I contact for credit analysis?"},
    {"role": "user", "content": "Tell me about David Clark."},
    # Add more user messages as needed
]

# Initial conversation with the system message and user messages
conversation = [system_message] + user_messages

# Initialize the variable to control the loop
quit_chat = False

while not quit_chat:
    # Get user input
    user_input = input("User: ")

    # Check if the user wants to quit
    if user_input.lower() == 'quit':
        quit_chat = True
        continue

    # Add user input to the conversation
    user_messages.append({"role": "user", "content": user_input})
    conversation = [system_message] + user_messages

    # Get the completion from OpenAI
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    # Display assistant's response
    assistant_response = completion.choices[0].message.content
    print("Assistant:", assistant_response)

    # Update the conversation with the assistant's response
    user_messages.append({"role": "assistant", "content": assistant_response})
    conversation = [system_message] + user_messages

    # Display information about the completion
    print("ID:", completion.id)
    print("Finish Reason:", completion.choices[0].finish_reason)
    print("Index:", completion.choices[0].index)
    print("Completion Tokens:", completion.usage.completion_tokens)
    print("Prompt Tokens:", completion.usage.prompt_tokens)
    print("Total Tokens:", completion.usage.total_tokens)
    print("\n")

# End of the chat loop
print("Chatbot session ended.")
