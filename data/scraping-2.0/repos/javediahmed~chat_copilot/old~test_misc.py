"""
Public Version ChatGPT Browser Prototype

Date: 5/25/2023
Author: Javed I. Ahmed; chat@aicanalytics.com
Dependencies: openai, os, json, datetime.datetime
Licence: Please cite if used, pls use if cited.
"""

import openai
import os
import json
from datetime import datetime

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

    def prompt_user(self, message):
        """
        Prompt the user for input with the given message.

        Args:
            message (str): The prompt message.

        Returns:
            str: User input.
        """
        return input(f"{message}\n").strip()

    def check_api_key(self):
        """
        Check the API key and set it for the OpenAI library.
        """
        while not self.api_key:
            print("API Key not found.")
            self.api_key = self.prompt_user("Please enter your OpenAI API Key: ")
        openai.api_key = self.api_key

    def chat(self, copilot=False):
        """
        Perform the chat interaction with the GPT model.
        """
        self.check_api_key()
        model_name = self.settings['Model']
        model_value = MODELS[model_name]
        header = "Copilot" if copilot else "Chatting"
        print(f'\n{header} with {model_name} ({model_value})')

        while True:
            query = self.prompt_user("\nEnter your query ('f' to submit by file or 'x' to exit): ")
            if query.lower() == 'f':
                json_query = self.prompt_user("Enter the path to JSON file: ")
                try:
                    with open(json_query, 'r') as file:
                        data = json.load(file)
                        query = data['query']
                except (json.JSONDecodeError, KeyError, FileNotFoundError):
                    print("Invalid JSON query. Please try again.")
                    continue
            elif query.lower() == 'x':
                break
            role = self.settings["Query Settings"]["Role"]
            prompt = f"{role}: {query}"
            query_settings = self.settings["Query Settings"]
            try:
                response = openai.Completion.create(
                    engine=model_value,
                    prompt=prompt,
                    max_tokens=query_settings["Max Tokens"],
                    temperature=query_settings["Temperature"]
                )
                response_text = response.choices[0].text.strip()
                self.history.append({"query": query, "response": response_text, "copilot_response": response_text if copilot else ""})
                print(f"\n{response_text}")
            except openai.error.APIError as e:
                print(f"\nOpenAI API Error: {e}")
                self.api_key = ""
                self.check_api_key()

    def update_settings(self):
        """
        Update the chat settings.
        """
        print("\nWhich setting would you like to change?")
        self.display_settings(self.settings)
        self.change_settings(self.settings)

    def display_settings(self, settings, indices=None):
        """
        Display the current chat settings.

        Args:
            settings (dict): The chat settings.
            indices (list): A list of indices for the settings.
        """
        if indices is None:
            indices = []

        for key, value in settings.items():
            if key != "Menu":
                indices.append(str(len(indices) + 1))

                if isinstance(value, dict):
                    print(f"{'. '.join(indices)}. {key}:")
                    self.display_settings(value, indices)
                else:
                    print(f"{'. '.join(indices)}. {key}: {value}")
                if not isinstance(value, dict):
                    indices.pop()

    def display_settings(self, settings, indices=None):
        """
        Display the current chat settings.

        Args:
            settings (dict): The chat settings.
            indices (list): A list of indices for the settings.
        """
        if indices is None:
            indices = []

        for key, value in settings.items():
            if key != "Menu":
                indices.append(str(len(indices) + 1))

                if isinstance(value, dict):
                    print(f"{'. '.join(indices)}. {key}:")
                    self.display_settings(value, indices)
                else:
                    print(f"{'. '.join(indices)}. {key}: {value}")
                if not isinstance(value, dict):
                    indices.pop()

    def update_subsettings(self, settings):
        """
        Update the subsettings.

        Args:
            settings (dict): The subsettings.
        """
        print("\nWhich sub-setting would you like to change?")
        self.display_settings(settings)
        self.change_settings(settings)

    def export_data(self):
        """
        Export the chat history to a file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpt_chat_export_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(self.history, f)
        print(f"\nChat history exported to {filename}")

    def display_help(self):
        """
        Display the help text.
        """
        print("""
        1. Chat: Initiate a chat with the selected GPT model.
        2. Copilot: Get code suggestions from the selected GPT model.
        3. Export data: Save the chat history to a file.
        s. Settings: Change the chat settings.
        ?. Help: Display this help text.
        x. Exit: Quit the application.
        
        You can also type "copilot::" to get code suggestions from the selected GPT model at any time.
        """)

    def handle_menu_choice(self, choice):
        """
        Handle the user's menu choice.

        Args:
            choice (str): The user's menu choice.
        """
        if choice == "1":
            self.chat()
        elif choice == "2" or choice == "copilot::":
            self.chat(copilot=True)
        elif choice == "3":
            self.export_data()
        elif choice == "s":
            self.update_settings()
        elif choice == "?":
            self.display_help()
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
            user_choice = self.prompt_user("Enter your choice: ")
            self.handle_menu_choice(user_choice)

if __name__ == "__main__":
    chat_gpt = ChatGPT()
    chat_gpt.run()
