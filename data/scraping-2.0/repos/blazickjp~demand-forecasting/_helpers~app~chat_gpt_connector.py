import openai
from config import OPENAI_API_KEY


class ChatGPTConnector:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.conversation_history = []

    def get_response(self, message):
        # Append the user's message to the conversation history
        self.conversation_history.append({"role": "user", "content": message})

        # Generate a response using OpenAI's chat models
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can change to a different chat model if available
            messages=self.conversation_history,
            max_tokens=100,  # You can adjust max tokens based on your requirement
        )

        # Append the model's response to the conversation history
        bot_response = response["choices"][0]["message"]["content"]
        self.conversation_history.append({"role": "assistant", "content": bot_response})

        return bot_response
