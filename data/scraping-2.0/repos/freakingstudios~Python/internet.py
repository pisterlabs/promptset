import openai
import pyttsx3

# Set up your OpenAI API key
openai.api_key = "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"

def generate_response(messages):
    # Set up the OpenAI Chat API parameters
    model = "gpt-3.5-turbo"

    # Generate the response using the Chat API
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    # Extract and return the assistant's reply
    reply = response.choices[0].message['content']
    return reply

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Initialize the conversation
conversation = [
    {"role": "system", "content": "You are a helpful assistant that provides details."},
    {"role": "user", "content": ""}
]

# Start the conversation loop
while True:
    # Get user input
    user_input = input("User: ")

    # Set user input in the conversation
    conversation[-1]["content"] = user_input

    # Check if the user wants to stop the conversation
    if user_input.lower() in ["stop", "quit"]:
        break

    # Generate response using OpenAI
    assistant_response = generate_response(conversation)

    # Add assistant response to the conversation
    conversation.append({"role": "assistant", "content": assistant_response})

    # Speak out the assistant's response
    speak(assistant_response)

    # Print the assistant's response
    print("Assistant:", assistant_response)
