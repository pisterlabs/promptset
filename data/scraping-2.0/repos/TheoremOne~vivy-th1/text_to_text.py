import openai
import logging
from constants import OPENAI_API_KEY

logger = logging.getLogger(__name__)

class TextToText:
    def __init__(self, messages):
        openai.api_key = OPENAI_API_KEY
        self.messages = messages
        self.token_count = self.count_tokens(self.messages)
    
    def count_tokens(self, messages):
        count = 0
        for message in messages:
            count += len(message["content"]) // 4
        return count

    def generate_response(self, user_input):
        try:
            role = "user" if self.messages[-1]["role"] == "assistant" else "assistant"
            self.messages.append({"role": role, "content": user_input})

            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages)
            response = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response})
            while self.count_tokens(self.messages) > 4000 and len(self.messages) >= 3:
                self.messages = [self.messages[0]] + self.messages[2:]
            self.token_count = self.count_tokens(self.messages)

            return response
        except Exception as e:
            logger.error(f"Error in TextToText: {e}")
            raise
        