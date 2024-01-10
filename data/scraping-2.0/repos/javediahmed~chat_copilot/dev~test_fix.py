import openai
import os
import json
from datetime import datetime
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODELS = {
    "GPT-1": "text-gpt-1-en-12b",
    "GPT-2": "text-gpt-2-en-117b",
    "GPT-3": "text-davinci-002",
    "GPT-3.5": "text-davinci-003",
    "GPT-4": "text-davinci-004",
    "Jurassic-1 Jumbo": "text-jurassic-1-jumbo-en-175b",
    "Megatron-Turing NLG": "text-megatron-turing-nlg-345m-355b",
    "WuDao 2.0": "text-wudao-2-0-en-1.76T"
}

DEFAULT_SETTINGS = {
    "Model": "GPT-3",
    "Query Settings": {
        "Max Tokens": 60,
        "Temperature": 0.5,
        "Role": "user"
    },
    "Menu": {
        "1": "Chat",
        "2": "Copilot",
        "3": "Export data",
        "s": "Settings",
        "?": "Help",
        "x": "Exit"
    }
}

class ChatGPT:
    """
    ChatGPT class for interacting with the OpenAI GPT models.
    """

    def __init__(self):
        """
        Initialize the ChatGPT instance.
        """
        self.api_key = OPENAI_API_KEY
        self.settings = DEFAULT_SETTINGS.copy()
        self.history = []
        self.session = PromptSession()

    def prompt_user(self, message):
        """
        Prompt the user for input with the given message.

        Args:
            message (str): The prompt message.

        Returns:
            str: User input.
        """
        return self.session.prompt(message + '\n')

    def check_api_key(self):
        """
        Check the API key and set it for the OpenAI library.
        """
        while not self.api_key:
            print("API Key not found.")
            self.api_key = self.prompt_user("Please enter your OpenAI API Key: ")
        openai.api_key = self.api_key

    # Remaining methods...
    def handle_menu_choice(self, choice):
        """
        Handle the user's menu choice.

        Args:
            user_choice (str): The user's menu choice.
        """
        if choice == "1":
            self.chat()
        elif choice == "2":
            self.chat(copilot=True)
        elif choice == "3":
            self.export_data()
        elif choice == "s":
            self.update_settings()
        elif choice == "x":
            print("Exiting the program...")
            exit()
        else:
            print("Invalid choice. Please try again.")


    def run(self):
        """
        Run the ChatGPT interface.

        Args:
            menu (dict): The menu options.
        """
        print("ChatGPT Browser Prototype")
        print(f"Date: {datetime.today().strftime('%Y-%m-%d')}")
        while True:
            print("\nMain Menu:")
            for key, value in self.settings["Menu"].items():
                print(f"{key}. {value}")
            user_choice = self.session.prompt("Enter your choice: ", completer=WordCompleter(list(self.settings["Menu"].keys()), ignore_case=True))
            self.handle_menu_choice(user_choice)

if __name__ == "__main__":
    chat_gpt = ChatGPT()
    chat_gpt.run()
