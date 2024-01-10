import openai
from cryptography.fernet import Fernet
from dotenv import load_dotenv, find_dotenv
from os.path import exists
import os

# These variables are used to store the API key and the path to the .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, 'cli_chat.env')

# This variable is used to store .gitignore file path
gitignore_path = os.path.join(script_dir, '.gitignore')

# Messages
initial_message = 'Control + C, --quit or -q to quit, --help or -h for show help message'
gitignore_message = '\033[31m' + 'WARNING!: This application generates cli_chat.env file which CONTAINS YOUR OPENAI API KEY information.\nCreating .gitignore file is recommended to ignore cli_chat.env and cli_chat.py (cli_chat itself)\nPlease type --gitignore or -gi to create .gitignore file for cli_chat.env and cli_chat.py or update existing file' + '\033[0m'
chatmode_message = 'Control + C, --quit or -q to quit, --help or -h for show help message, --new or -n to start a clear chat and start a new session'
nosave_hint_message = '\033[32m' + 'If you do not want to save OpenAI API Key in cli_chat.env\nYou can use --nosave or -ns command to prevent API Key save' + '\033[0m'
nosave_enabled_message = '\033[32m' + 'nosave setting (Never API Key Save) is enabled.\nAPI Key will not be saved in cli_chat.env' + '\033[0m'

# This is a list of help messages that are displayed when the user enters --help in API mode or chat mode
apimode_help_messages = [
    "'--help' or '-h': Show this help message",
    "'--gitignore' or '-gi': To create .gitignore file for cli_chat.env and cli_chat.py",
    "'--quit' or '-q': To quit the program",
    ]

chatmode_help_messages = [
    "'--help' or '-h': Show this help message",
    "'--gitignore' or '-gi': To create .gitignore file for cli_chat.env and cli_chat.py",
    "'--quit' or '-q': To quit the program",
    "'--new' or '-n': To start a new session",
    ]

# This function creates .gitignore file to ignore cli_chat.env and cli_chat.py
def create_gitignore():
    if not exists(gitignore_path):
        with open(gitignore_path, 'w') as f:
            f.write('.gitignore\n')
            f.write('cli_chat.env\n')
            f.write('cli_chat.py\n')
            print(".gitignore file is created")
    else:
        user_input = input('.gitignore file already exist in the same directory as cli_chat.py exists.\nEspecially, cil_chat.env contains your OpenAI API Key information.\nDo you want to add cli_chat.py and cli_chat.env file to it? [y/n]: ')
        if user_input == 'y':
            with open(gitignore_path, 'a') as f:
                f.write('\n')
                f.write('cli_chat.env\n')
                f.write('cli_chat.py\n')
                print(".gitignore file is updated to ignore cli_chat.env and cli_chat.py")
        elif user_input == 'n':
            print('Aborted')

# This function creates cli_chat.env file to the nosave setting or already exists cli_chat.env file to add nosave setting
def env_enable_nosave():
    # Create cli_chat.env file if it does not exist
    if not exists(env_path):
        with open(env_path, 'w') as f:
            f.write('nosave=True\n')
            print('OpenAI API Key saving feature is now disabled (nosave is enabled)')
    # If cli_chat.env already exists, it will add nosave=True to the end of the file
    # The last line of cli_chat.env if it exists, is always the new line character
    elif exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()
        # If nosave setting is already enabled, it will print a message
        if 'nosave=True\n' in lines:
            print('OpenAI API Key saving feature is already disabled (nosave is enabled)')
        # If cli_chat.env exists and nosave setting is not enabled, it will add nosave=True to the end of the file
        elif 'nosave=True\n' not in lines:
            with open(env_path, 'a') as f:
                f.write('nosave=True\n')
                print('OpenAI API Key saving feature is now disabled (nosave is enabled)')

# This function checks if the nosave setting is enabled
def nosave_check():
    # If cli_chat.env exists, it will check if nosave setting is enabled
    if exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()
            # If nosave setting is enabled, it will return True
            if 'nosave=True\n' in lines:
                return True
            # If nosave setting is not enabled, it will return False
            elif 'nosave=True\n' not in lines:
                return False
    # If cli_chat.env does not exist, it will return False
    elif not exists(env_path):
        return False

# This function removes nosave setting from cli_chat.env
def env_enable_save():
    # If cli_chat.env exists, it will remove nosave=True from the file
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.read()
        if "nosave=True\n" in lines:
            replaced_data = lines.replace('nosave=True\n', '')
            with open(env_path, 'w') as f:
                f.write(replaced_data)
                print("OpenAI API Key saving feature is now enabled (save is enabled)")
    elif not os.path.exists(env_path):
        print("OpenAI API Key saving feature is already enabled (save is enabled)")

# This function removes cli_chat.env file and replace with nosave=True only cli_chat.env file when nosave setting is enabled in Chat mode
def chatmode_nosave():
    # If cli_chat.env exists, it will remove cli_chat.env and create new cli_chat.env file with nosave=True
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.read()
        if "nosave=True\n" in lines:
            os.remove(env_path)
            with open(env_path, 'w') as f:
                f.write('nosave=True\n')
                print("Already existing cli_chat.env that contains OpenAI API related data has already deleted and replaced with a new file that contains only nosave=True.\nOpenAI API Key saving feature is now disabled (nosave is enabled)")
    # If cli_chat.env does not exist, it will create new cli_chat.env file with nosave=True
    elif not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write('nosave=True\n')
            print("OpenAI API Key saving feature is now disabled (nosave is enabled)")

# This function displays the CLI CHAT logo and messages
def aa():
    from art import text2art
    ascii_art = text2art('CLI CHAT')
    print(ascii_art)
    print('Control + C, --quit or -q to quit, --help or -h for show help message')
    if not exists(gitignore_path):
        print(gitignore_message)
    if nosave_check() == False:
        print(nosave_hint_message)
    # True means that the nosave setting is enabled
    else:
        print(nosave_enabled_message)

# This function provides a input for the user to enter their API key
def get_api_key():
    api_key_input = input('Please enter your OpenAI API key: ')
    if api_key_input == "--quit" or api_key_input == "-q":
        print('Bye')
        raise SystemExit
    elif api_key_input == "--help" or api_key_input == "-h":
        for help_message in apimode_help_messages:
            print(help_message)
        return get_api_key()
    elif api_key_input == "--gitignore" or api_key_input == "-gi":
        create_gitignore()
        return get_api_key()
    elif api_key_input == "--save" or api_key_input == "-s":
        if nosave_check() == True:
            env_enable_save()
        elif nosave_check() == False:
            print('OpenAI API Key saving feature is already enabled')
        return get_api_key()
    elif api_key_input == "--nosave" or api_key_input == "-ns":
        if nosave_check() == True:
            print('OpenAI API Key saving feature is already disabled')
        elif nosave_check() == False:
            env_enable_nosave()
        return get_api_key()
    elif api_key_input == "":
        print('Please enter your OpenAI API key')
        return get_api_key()
    else:
        return api_key_input

# API Key AES encryption
# AES Key Generator
def generate_key():
    return Fernet.generate_key()

# This function encrypts the API key
def encrypt_message(message, key):
    f = Fernet(key)
    encrypted_message = f.encrypt(message)
    return encrypted_message

# This function decrypts the API key
def decrypt_message(encrypted_message, key):
    f = Fernet(key)
    decrypted_message = f.decrypt(encrypted_message)
    return decrypted_message

# This function encrypts and saves the API key
def save_api_key(api_key):
    key = generate_key()  
    encrypted_api_key = encrypt_message(api_key.encode(), key)

    with open(env_path, 'w') as f:
        f.write(f'EAK_cd6cfa29eba6af74cb323d7ba357={encrypted_api_key.decode()}\n')
        f.write(f'K_17fa993d5eecbd361f30baf0b9b2={key.decode()}\n')


# This function decrypts and loads the API key from cli_chat.env
def load_api_key():
    print('Loading...')
    load_dotenv(find_dotenv(filename='cli_chat.env'))
    encrypted_api_key = os.getenv('EAK_cd6cfa29eba6af74cb323d7ba357')
    key = os.getenv('K_17fa993d5eecbd361f30baf0b9b2')

    if encrypted_api_key and key:
        api_key = decrypt_message(encrypted_api_key.encode(), key.encode())
        return api_key.decode()
    else:
        return None

# This function validates the API key
def validate_api_key(api_key):
    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="test",
            max_tokens=5
        )
        print('API key is valid.')
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        print("API key is not valid. Please check your key and try again.")
        return False

# This function checks the availability of the GPT-4 model
def check_model_availability(api_key, model_name, default_model_name):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model = model_name,
            messages = [({'role': 'user', 'content': 'test'})],
        )
        print(f"{model_name} is available.")
        return True
    except Exception as e:
        print(f"{model_name} is not available, using {default_model_name} instead.")
        print(f"({e})")
        return False

# This function creates a input for the user to enter their prompt
def get_prompt():
    return input('Please enter your prompt: ')

# This function prints the message before chat mode
def print_message_before_chatmode():
    return print(chatmode_message)

# This is the main function
def main():
    chat = []
    aa()
    try:
        api_key = load_api_key()

        if not api_key:
            api_key = get_api_key()
        
        while not validate_api_key(api_key):
            api_key = get_api_key()

        if not nosave_check():
            save_api_key(api_key)

        if check_model_availability(api_key, "gpt-4", "gpt-3.5-turbo"):
            model_name = "gpt-4"
        else:
            model_name = "gpt-3.5-turbo"

        print_message_before_chatmode()

        while True:
            prompt = get_prompt()
            user_prompt = {'role': 'user', 'content': prompt}
            chat.append(user_prompt)

            if prompt == "":
                continue

            if prompt == "--help":
                for help_message in chatmode_help_messages:
                    print(help_message)
                continue

            if prompt == "--quit":
                raise KeyboardInterrupt

            if prompt == "--gitignore" or prompt == "--gi":
                create_gitignore()
                continue

            if prompt == "--save" or prompt == "-s":
                if nosave_check() == True:
                    env_enable_save()
                elif nosave_check() == False:
                    print('OpenAI API Key saving feature is already enabled')
                continue
            
            elif prompt == "--nosave" or prompt == "-ns":
                if nosave_check() == True:
                    print('OpenAI API Key saving feature is already disabled')
                elif nosave_check() == False:
                    env_enable_nosave()
                    chatmode_nosave()
                    break

            # if prompt == "--nosave" or prompt == "--ns":
            #     if not nosave_check():
            #         env_enable_nosave()
            #     else:
            #         print("OpenAI API Key save feature is already disabled.")
            #     continue

            if prompt == "--new":
                chat = []
                print("Chat history cleared, starting a new chat.")
                continue

            # Error handling
            try:  
                print("Chat:")
                response = openai.ChatCompletion.create(
                    model = model_name,
                    messages = chat,
                )
                assistant_response = response.choices[0].message.content
                print(assistant_response)
                assistant_prompt = {'role': 'assistant', 'content': assistant_response}
                chat.append(assistant_prompt)
            except Exception as e:
                print(f"Error: {e}")
    # Keyboard interrupt handling
    except KeyboardInterrupt:
        print("Bye")

if __name__ == "__main__":
    main()


"""
This application is created by: Shigeki N (Discord: mondayfrenzy)

This application is created using OpenAI API and Python.

Description:
This application is a CLI chatbot that uses OpenAI API (GPT-3.5-turbo or GPT-4) to generate responses.
It facilitates developer experience by allowing developers to chat with the chatbot directly from the terminal.
I (personally) highly recommend using this application in VSCode's, JetBrain's or some IDE integrated terminal.

Commands:
** Global Commands **
--help or --h: Show help message
--quit or --q: Quit the application
--gitignore or --gi: Create a .gitignore file
--save or --s: Enable the OpenAI API Key save feature
--nosave or --ns: Do not save the API key in the cli_chat.env file
** Effective in Chat Prompt **
--new or --n: Start a new chat

WARNING:
This application generates a cli_chat.env file in the same directory as the application.
This file contains the encrypted OpenAI API KEY.
DO NOT SHARE THIS FILE WITH ANYONE.
DO NOT UPLOAD THIS FILE TO GITHUB OR ANY OTHER PLATFORM.
To prevent this file from being uploaded to GitHub, run the following command in the application cli:
--gitignore or --gi
This command will create a .gitignore file to ignore the cli_chat.env and cli_chat.py file in the same directory.
Even if you already have a .gitignore file in the same directory, this command will add the cli_chat.env and cli_chat.py file to the .gitignore file.

Donations:
ETH Address: 0xE06174d2fa3b30f17746Db3C03898f08C1b18DDc

Note:
This application creates an environment file called cli_chat.env in the same directory as the application.
This file contains the encrypted API key and the encryption key.
If you want to delete the API key stored, delete the cli_chat.env file.

Notice:
This application is not affiliated with OpenAI in any way.
"""
# Apache License 2.0 © 2023 Shigeki N