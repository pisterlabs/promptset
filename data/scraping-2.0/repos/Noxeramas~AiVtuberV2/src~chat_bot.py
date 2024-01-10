import openai
from openai import OpenAI
import request_manager
import config_paths
import random

from config_manager import ConfigManager

config_file_path = config_paths.get_config_file_path()
config = ConfigManager(config_file_path)

lore_paths = {
    "bot1": config_paths.get_bot1_lore_file_path(),
    "bot2": config_paths.get_bot2_lore_file_path()
}

openai_api_key = config.get_setting(['api_keys', 'openai'], "")
client = OpenAI(api_key=openai_api_key)

MAX_TOKENS = 4096


class ChatBotManager:
    def __init__(self, lore_file_paths):
        self.model_choice = "GPT"
        self.openai_history = ""
        self.ooba_history = {'internal': [], 'visible': []}
        self.lores = {}
        self.config = config.load_settings()
        # Load lore for each bot
        for bot, path in lore_file_paths.items():
            with open(path, 'r', encoding='utf-8') as f:
                self.lores[bot] = f.read()

    def chat_bot(self, user_input, bot_id, interaction_type="user"):
        """
        :param interaction_type:
        :param user_input:
        :param bot_id:
        :return text_response:

        Generate a response from the chatbot based on user input and bot identity.
        """
        if self.model_choice == "GPT":
            # Prepare the request for the GPT model
            openai_data = request_manager.OpenAiRequestManager()
            request = openai_data.request
            if openai_api_key == "":
                return "Please enter an OpenAI API key in the settings tab."
            # Update the chat history with the latest user input
            # TODO: This does not work current with openai, i cannot get the history to be acknowledged
            # self.openai_history += f"\nUser says: {user_input}"
            # Check if we need to truncate the history to stay within the token limit
            # self.truncate_history_if_needed(MAX_TOKENS)

            # Create a system message with lore and history
            lore_system_message = {"role": "system", "content": self.lores[bot_id]}
            # chat_history_system_message = {"role": "system", "content": self.openai_history[bot_id]}

            message_prefix = f"{bot_id.capitalize()} says:" if interaction_type == "bot" else "User says:"

            # Prepare the messages array
            messages = [
                lore_system_message,
                # chat_history_system_message,
                {"role": "user", "content": f"{message_prefix} {user_input}"}
            ]

            response = ""
            # Make the API call
            try:
                response = client.chat.completions.create(
                    model=request["model"],
                    messages=messages
                )
            except openai.APIConnectionError as e:
                # Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                pass
            except openai.APIError as e:
                # Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                pass
            except openai.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                pass

            # Process the response into text
            text_response = response.choices[0].message.content

            # Update the chat history for this bot
            # self.openai_history[bot_id] += f"\n{message_prefix} {text_response}"
            # Check if we need to truncate the history to stay within the token limit
            # self.truncate_history_if_needed(bot_id, MAX_TOKENS)
            print(f"{bot_id.capitalize()}: {text_response}")
            return text_response

    def initiate_conversation(self):
        """
        Initiates a conversation between the two bots for a specified number of turns.
        """

        bot_names = {
            "bot1": self.config["bot1_name"],
            "bot2": self.config["bot2_name"]
        }

        # Initialize an empty string to keep track of the conversation history
        conversation_history = ""

        # Randomly choose which bot starts the conversation
        next_speaker_key = random.choice(["bot1", "bot2"])

        # Set up a scenario for the bots to interact
        conversation_context = (
            f"{self.config['bot1_name']} and {self.config['bot2_name']} are having a conversation."
        )

        # First bot starts the conversation
        starter_prompt = f"{bot_names[next_speaker_key]} starts: 'A twitch user just said that they like your art, what do you say?'"
        print(f"Starter Prompt: {starter_prompt}")
        print(starter_prompt)

        # Process the starter response
        response = self.chat_bot(conversation_context + " " + starter_prompt, next_speaker_key, interaction_type="bot")
        print(response)
        conversation_history += f" {starter_prompt} {response}"

        # Alternate between bots for the remaining turns
        for _ in range(random.randint(3, 5) - 1):
            next_speaker_key = "bot1" if next_speaker_key == "bot2" else "bot2"
            prompt = f"{bot_names[next_speaker_key]} responds: '{response}'"
            response = self.chat_bot(conversation_context + conversation_history + prompt, next_speaker_key,
                                     interaction_type="bot")
            print(response)

    def truncate_history_if_needed(self, max_tokens):
        # Estimate the number of tokens for a given string
        def estimate_tokens(message):
            return len(message) // 4

        if self.model_choice == 'GPT':
            # Calculate the total token count of the current history
            total_tokens = estimate_tokens(self.openai_history)

            # Truncate the history if the total token count exceeds max_tokens
            while total_tokens > max_tokens:
                # Find the first newline character and remove up to that point
                newline_index = self.openai_history.find('\n')
                if newline_index == -1:
                    # If no newline character, clear the history
                    self.openai_history = ""
                    break
                else:
                    # Remove the oldest part of the history up to the newline
                    self.openai_history = self.openai_history[newline_index + 1:]
                    total_tokens = estimate_tokens(self.openai_history)
