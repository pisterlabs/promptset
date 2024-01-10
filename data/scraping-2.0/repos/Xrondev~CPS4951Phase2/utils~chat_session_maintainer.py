"""
The Chat GPT need a chat history to generate a response with 'memory'.
This class is used to maintain the chat history and maintain the initial prompt.
"""
import openai

from utils.configuration import Configuration
from utils.recorder import Recorder

config = Configuration()


class ChatSessionMaintainer:
    """
    csm for short. A chat history maintainer
    """

    def __init__(self):
        openai.proxy = config.get('App', 'proxy')
        openai.api_key = config.get('User', 'api_key')
        self.messages_history = []
        self.recorder = Recorder()

    def chat(self, message, emotion_dict: dict) -> str:
        """
        Chat with the GPT model
        :param message: the user's input
        :param emotion_dict:
        the emotion likelihood dict, will be provided to the GPT model to generate a response

        :return:
        the response from the GPT.Or an error message if the API key is invalid/ quota exceeded.
        """
        if len(self.messages_history) == 0:
            # System prompt for ChatGPT
            self.messages_history.append({
                "role": "system",
                "content": f"{config.get('User', 'prompt')} \n "
                           f"user's emotion likelihood dict is: {str(emotion_dict)}"
            })

        try:
            self.messages_history.append({
                "role": "user",
                "content": f"{message} \n user's emotion likelihood dict is: {str(emotion_dict)}"
            })

            completion = openai.ChatCompletion.create(
                model=config.get('App', 'model'),
                messages=self.messages_history,
            )

            self.messages_history.append({
                "role": "assistant",
                "content": completion.choices[0].message.content
            })

            self.recorder.record_token(completion)  # record the token count
            self.recorder.record_chat(completion, self.messages_history)  # record the chat history

            return completion.choices[0].message.content
        except openai.error.AuthenticationError:
            return "Invalid API Key, check your API key in the settings"
        except openai.error.RateLimitError:
            return "Exceeded current quota, check your API plan and billing details"

    def clear(self) -> None:
        """
        Clear the chat history
        :return: None
        """
        self.messages_history = []

    def print_session(self) -> list:
        """
        Print the chat history
        :return: A list of messages
        """
        return self.messages_history
