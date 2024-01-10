from openai import OpenAI

client = OpenAI(
    api_key="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"
)

# Initialize the chat messages history
messages = [
    {"role": "system", "content": "You are a helpful assistant with extensive experience in business management."},
    {"role": "assistant", "content": "How can I help?"}
]

# Function to display the chat history
def display_chat_history(messages):
    for message in messages:
        print(f"{message['role'].capitalize()}: {message['content']}")

# Function to get the assistant's response
def get_assistant_response(messages):
    r = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0613:nazarbayev-university::8JlmdSDJ",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )
    response = r.choices[0].message.content
    return response

# Main chat loop
#while True:
# Display chat history
#display_chat_history(messages)

# Get user input
prompt = input("User: ")
messages.append({"role": "user", "content": prompt})

# Get assistant response
response = get_assistant_response(messages)
messages.append({"role": "assistant", "content": response})