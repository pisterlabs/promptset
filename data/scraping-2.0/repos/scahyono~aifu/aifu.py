from openai import OpenAI
import logging
from dotenv import load_dotenv
import shutil
import os

# Load environment variables from .env file
load_dotenv()

# Configuration Management: Load API key from environment variables
api_key = os.environ.get('OPENAI_API_KEY')
MODEL="gpt-4-1106-preview" # Update this to the desired model version
TOKEN_TARGET = 100000 # Increase if the bot is forgetful

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Check if the aifus folder is empty
if not os.listdir('aifus'):
    # If empty, copy each subdirectory from aifus_examples to aifus
    for item in os.listdir('aifus_examples'):
        s = os.path.join('aifus_examples', item)
        d = os.path.join('aifus', item)
        if os.path.isdir(s):
            shutil.copytree(s, d, False, None)

# List the content of the aifus folder
aifus = next(os.walk('aifus'))[1]

# If only one AIfu, choose it automatically, else ask the user to choose an AIfu
if len(aifus) == 1:
    aifu_location = aifus[0]
else:
    print("Choose an AIfu to chat with:")
    for index, aifu in enumerate(aifus):
        print(f"{index + 1}. {aifu.replace('_', ' ').title()}")

    choice = int(input("Enter the number of your choice: "))
    aifu_location = aifus[choice - 1]

# Set the aifu as the camel case of the aifu_location
aifu = aifu_location.replace("_", " ").title()

# Set up logging with timestamp
logging.basicConfig(filename=f'aifus/{aifu_location}/aifu_chat.log', level=logging.INFO, format='%(asctime)s %(message)s')

# File to store the conversation history
history_filename = f'aifus/{aifu_location}/conversation_history.txt'

def compress_text(text):
    """
    Compress the text using OpenAI API.
    :param text: The text to compress.
    :return: The compressed text.
    """
    if not text:
        return ""  # Return an empty string if the text is empty

    # Check the length of the conversation history
    if len(text.split()) < TOKEN_TARGET:
        return text

    response = client.chat.completions.create(
        model=MODEL,
        prompt=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Make {TOKEN_TARGET/2} words summary of the following text:\n\n{text}"}
        ]
    )
    compressed_text = response.choices[0].message.content.strip()
    return compressed_text

def aifu_response(user_input):

    # Load the conversation history from file
    conversation_history = load_conversation_history()

    try:
        # Connect to the OpenAI API and get a response using the updated method
        with open(f'aifus/{aifu_location}/persona.txt', 'r', encoding='utf-8') as file:
            aifu_persona = file.read()
        instruction = aifu_persona + " Replace 'As an AI,' with 'As an AIfu'."
        if user_input == "":
            instruction += " Introduce yourself and ask for my name."

        response = client.chat.completions.create(
            stream=True,
            model=MODEL,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": conversation_history},
                {"role": "user", "content": user_input}
            ]
        )

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return f"An error occurred: {e}"

    return response

def log_conversation(user_input, response_content):
    """
    Log the conversation and update the conversation history.
    :param user_input: The user's input.
    :param response_content: The response from AIfu.
    """
    # Log the conversation
    logging.info(f"I: {user_input} | You: {response_content}")

    # Load the conversation history from file
    conversation_history = load_conversation_history()

    # Add the user's message to the conversation history
    conversation_history += f" | I: {user_input} | You: {response_content}"

    # Write the updated conversation history back to the file
    with open(history_filename, 'w', encoding='utf-8') as file:
        file.write(conversation_history)

def chat():
    print("Hello! Type 'bye' to exit.")

    conversation_history = load_conversation_history()
    # if conversation history file is empty then ask for introuction otherwise continue the last conversation
    if conversation_history == "":
        stream = aifu_response("")
    else:
        stream = aifu_response("I'm back!")
    printResponse(stream)

    while True:
        user_input = input("\033[92mYou: \033[0m")  # \033[92m is the ANSI escape code for green text, \033[0m resets the text color
        if user_input.lower() in ('bye', 'exit', 'quit'):
            print(f"\033[94m{aifu}: \033[0mGoodbye! Have a nice day!")  # \033[94m is the ANSI escape code for blue text
            break
        stream = aifu_response(user_input)
        fullresponse=printResponse(stream)
        log_conversation(user_input, fullresponse)

def printResponse(response):
    print(f"\033[94m{aifu}: \033[0m", end='')
    fullresponse=""
    for chunk in response:
        chunkContent = chunk.choices[0].delta.content
        if chunkContent is None:
            chunkContent = "\n"
        fullresponse += chunkContent
        print(chunkContent, end='')
    return fullresponse

def load_conversation_history():
    try:
        with open(history_filename, 'r', encoding='utf-8') as file:
            conversation_history = file.read()
    except FileNotFoundError:
        conversation_history = ""

    # Compress the conversation history if it exceeds 4096 tokens
    conversation_history = compress_text(conversation_history)

    return conversation_history

chat()
