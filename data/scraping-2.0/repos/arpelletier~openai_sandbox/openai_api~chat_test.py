import os
import openai
import sys

sys.path.append("../")
from config import OPENAI_KEY

# Changed the open ai key here
openai.api_key = OPENAI_KEY

from utils.utils import get_project_root

def parse_message(chat_completion):

    message = chat_completion['choices'][0]['message']

    role = message['role'].capitalize()
    content = message['content']

    return "%s: %s"%(role,content)


def get_log_file(directory):
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Find the next available log file
        log_file = None
        i = 0
        while True:
            log_file = os.path.join(directory, f"log_{i}.txt")
            if not os.path.exists(log_file):
                break
            i += 1

        return log_file
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def write_to_log(log_file, text):
    try:
        with open(log_file, 'a') as file:
            file.write(text + '\n')
    except Exception as e:
        print(f"An error occured: {str(e)}")

def single_chat(user_input, timeout_threshold=100):
    # TODO: if takes longer than thresh then skip/rerun
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": user_input}])
    message = parse_message(chat_completion)
    # Write to log as well
    log_folder = os.path.join('../chat_log')
    log_file = get_log_file(log_folder)
    write_to_log(log_file, "User: "+ user_input)
    write_to_log(log_file, message)
    return message

def start_chat(log_file=None, text_file_input=False, text_file_path='query.txt'):
    first_pass = True
    while True:
        # Get user input
        if text_file_input and first_pass:
            first_pass = False
            with open(text_file_path) as f:
                user_input = '\n'.join(f.readlines())
            print("User: {}".format(user_input))
        else:
            user_input = input("User: ")
        
        print("Got the input.")

        # Send to API

        # Just have start chat call single chat
        # BUT instead of making a new chat every time just have it continue with the previous context
        # Boolean flag that returns whether
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": user_input}])
        response = parse_message(chat_completion)

        print(response)

        if log_file:
            write_to_log(log_file, "User: "+ user_input)
            write_to_log(log_file, response)

# Send output to log folder
if __name__ == "__main__":
    log_folder = os.path.join('../chat_log')
    log_file = get_log_file(log_folder)

    # Start chat
    if len(sys.argv) >= 2:
        start_chat(log_file, True)
    else:
        start_chat(log_file)

