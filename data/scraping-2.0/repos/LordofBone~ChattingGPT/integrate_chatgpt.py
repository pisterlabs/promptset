import logging

import openai

from .config.api_config import openai_api_key, default_model

logger = logging.getLogger(__name__)
logger.debug("Initialized")


class IntegrateChatGPT:
    def __init__(self, model=default_model, role="You are a helpful assistant", use_history=False):
        """
        Initialize the chatgpt integration class along with the api key.
        """
        openai.api_key = openai_api_key

        self.model = model

        self.context = role

        self.use_history = use_history

        self.conversation_history = [
            {"role": "system", "content": self.context}
        ]

        logging.debug(f"Initialized with model: {model}, role: {role}, use_history: {use_history}")

    def get_response(self, text_input):
        """
        Get the chatgpt response, using the context and text input set.
        If use_history is True, it will maintain a conversation history.
        :return:
        """
        logging.debug(f"Received input: {text_input}")
        if self.use_history:
            return self._get_chatgpt_response_with_history(text_input)
        else:
            return self._get_chatgpt_response(text_input)

    def _get_chatgpt_response(self, text_input):
        """
        Get the chatgpt response without using history.
        :return:
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": self.context},
                      {"role": "user", "content": text_input}]
        )

        assistant_message = response['choices'][0]['message']['content']
        logging.debug(f"Assistant response (without history): {assistant_message}")

        return assistant_message

    def _get_chatgpt_response_with_history(self, text_input):
        """
        Get the chatgpt response using history.
        :param text_input:
        :return:
        """
        # Add user's message to the history
        self.conversation_history.append({"role": "user", "content": text_input})
        logging.debug(f"Conversation history: {self.conversation_history}")

        # Get a response from OpenAI
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.conversation_history
        )

        # Extract the assistant's message from the response
        assistant_message = response['choices'][0]['message']['content']
        logging.debug(f"Assistant response (with history): {assistant_message}")

        # Add assistant's message to the history
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message
