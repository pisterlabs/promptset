import openai
from datetime import datetime
import os
import urllib.request
import pywhatkit
from users import user_manager


# API Key
api_key_path = './Api_keys/api_key_openai.txt'

# Read the API key from the text file
with open(api_key_path, 'r') as f:
    openai_api_key = f.read().strip()
openai.api_key = openai_api_key

# Define the path to the directory where the generated images will be saved
image_dir = "./Jarvis_Memory/images"
# Check if the directory exists, create it if it doesn't
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


# Generate response
def generate_response(user_input, conversation_history):
    system_content = "You are JARVIS (Just A Rather Very Intelligent System), respectively the household assistance of the "+user_manager.current_user.name+" family and designed by Mr. "+user_manager.current_user.name+" (as Jarvis, you call the user as Sir.). You are a helpful AI assistant and your purpose is to make human life better, with helpful answers."

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": conversation_history + "\\n" + user_manager.current_user.name + ": " + user_input}
        ],
        max_tokens=320,
    )

    message = response['choices'][0]['message']['content'].strip()
    conversation_history += "\\n" + user_manager.current_user.name + ": " + user_input + "\\n" + message
    return message, conversation_history






# Generate image
def generate_image(image_prompt):
    image_response = openai.Image.create(
        prompt=image_prompt,
        n=1,
        size="1024x1024"
        )
    image_url = image_response['data'][0]['url']

    return image_url

# Play music
def play_music(song):
    pywhatkit.playonyt(song)

# Get help
def print_help():
    help_message = '''
    Here are some tips and commands for using the chatbot:

    1. Type your questions or statements normally, and the chatbot will respond.
    2. To generate an image, type "generate image:" followed by a description, example: ("generate image: a beautiful sunset").
    3. Play anything from Youtube, use command: "play:"
    4. Search on Google with command: "search:"
    4. To exit the chat, type "exit" or "quit".
    
    Note: If the chatbot provides an unsatisfactory response, try rephrasing your question or statement.
    '''
    print(help_message)
