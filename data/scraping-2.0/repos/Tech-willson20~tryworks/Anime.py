import openai

# Set up OpenAI API key
openai.api_key = "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"

# Define the conversation history
messages = [
    {"role": "system", "content": "You are an AI assistant"},
]

# Define the chat function
def chat():
    while True:
        # Prompt the user for input
        user_input = input("You: ")

        # Add the user's input to the conversation history
        messages.append({"role": "user", "content": user_input})

        # Call the OpenAI API with the conversation history
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

        # Extract the assistant's reply from the API response
        assistant_reply = response["choices"][0]["message"]['content']

        # Add the assistant's reply to the conversation history
        messages.append({"role": "assistant", "content": assistant_reply})

        # Print the assistant's reply
        print("Ai Willson:", assistant_reply)

# Call the chat function to start the chatbot
chat()

