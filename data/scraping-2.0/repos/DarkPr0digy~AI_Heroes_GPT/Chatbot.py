import json
import openai
from InsightManager import InsightManager

MODEL_NAME = "gpt-3.5-turbo"
MODEL_TEMPERATURE = 0.9


class Chatbot:
    def __init__(self, name: str, user_total_characters=None, chatbot_total_words=None, conversation=None):
        """ Create a chatbot with a given personality
        :param name: Name of the personality
        :param user_total_characters: Total number of characters typed by the user
        :param chatbot_total_words: Total number of words used by the chatbot
        :param conversation: List of messages in the conversation
        """
        self.name = name
        self.introduction = None
        self.characteristic = None

        self.user_total_characters = 0 if user_total_characters is None else user_total_characters
        self.chatbot_total_words = 0 if chatbot_total_words is None else chatbot_total_words

        # Load API Key from Config File
        with open("config.json") as config_file:
            config = json.load(config_file)

        self.api_key = config["api_keys"]["open_ai"]
        openai.api_key = self.api_key

        self.messages = [] if conversation is None else conversation

        # Load Personality and Introduce Chatbot
        self._load_personality(name)

        if not self.messages:
            self._introduce()
        else:
            # Print Previous Conversation for user
            for message in self.messages:
                print(f"{self.name if message['role'] == 'assistant' else message['role']}: {message['content']}")
            self.messages.insert(0, {"role": "system", "content": self.characteristic})

    def _load_personality(self, personality_name: str):
        """ Load the personality from the personalities.json file
        :param personality_name: Name of the personality
        """

        with open("personalities.json") as personality_file:
            personalities = json.load(personality_file)

        personality_data = personalities.get(personality_name)

        if personality_data:
            self.introduction = personalities[personality_name]["starting_message"]
            self.characteristic = personalities[personality_name]["characteristic"]
        else:
            raise ValueError(f"Personality {personality_name} not found")

    def _introduce(self):
        """ Introduce the chatbot to the user
        """
        self.messages.extend([{"role": "system", "content": self.characteristic},
                              {"role": "assistant", "content": self.introduction}])
        print(f"{self.name}: {self.introduction}")

    def generate_response(self, user_input: str):
        """ Generate a response to the user input
        :param user_input: Input from the user
        :return: Response from the chatbot"""

        if user_input.lower() == "exit":
            InsightManager(self.api_key, self.name, self.user_total_characters, self.chatbot_total_words, self.messages)
            return "See you next time"

        self.messages.append({"role": "user", "content": user_input})

        conversation = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=self.messages,
            temperature=MODEL_TEMPERATURE)

        response = conversation.choices[0].message.content

        self.messages.append({"role": "assistant", "content": response})

        # Update Counts
        self.user_total_characters += len(user_input)
        self.chatbot_total_words += len(response.split(" "))

        return response
