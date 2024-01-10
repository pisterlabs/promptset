# Import prerequisite libraries
import os
import openai

# Setting the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the chat messages history
messages = [{"role": "assistant", "content": "How can I help?"}]

# Display chat history
def display_chat_history(messages):
    for message in messages:
        print(f"{message['role'].capitalize()}: {message['content']}")

# Get assistant response
def get_assistance_response(messages):
    r = openai.ChatCompletion.create(
        model = "gpt-4-1106-preview",
        messages = [{"role": m["role"], "content": m["content"]} for m in messages],
    )
    response = r.choices[0].message.content
    return response

# Main function - the chat loop
while True:
    # Display history
    display_chat_history(messages)

    # Get inputs
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})

    # Get response
    response = get_assistance_response(messages)
    messages.append({"role": "assistant", "content": response})