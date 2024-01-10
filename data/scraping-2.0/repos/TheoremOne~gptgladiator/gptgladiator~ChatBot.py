import openai
import tiktoken
from gptgladiator.GPTModel import GptModel


class ChatBot:
    def __init__(self, model: GptModel, temperature=1, messages=[], debug_output=False):
        """
        Initialize a `ChatBot` with a given `model` and `temperature`.

        This is a convenience abstraction over OpenAI's Chat API. It allows
        us to easily send messages to a chatbot and get responses back.
        """
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.debug_output = debug_output

    def get_completion(self, prompt):
        """
        Get a completion from the `model` using the given `prompt`.
        """
        self.messages.append({"role": "user", "content": prompt})

        current_tokens_used = self.count_total_tokens_in_messages(
            self.messages)
        response_tokens = self.model.tokens - current_tokens_used

        if self.debug_output:
            print(f"Token limit: {self.model.tokens}")
            print(f"Send Token Count: {current_tokens_used}")
            print(f"Tokens remaining for response: {response_tokens}")

        if response_tokens < 1000:
            raise ValueError(
                f"Too many tokens in the input. Max is {self.model.tokens} and you entered {current_tokens_used}. You must reserve 1000 tokens for the response yet you only have {response_tokens}. Remove content from your input and retry.")

        try:
            response = openai.ChatCompletion.create(
                model=self.model.name,
                temperature=self.temperature,
                messages=self.messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.debug_output:
                print("Error in chatbot: ", e)

    def count_total_tokens_in_messages(self, messages) -> int:
        """
        Count the total number of tokens in the given `messages`.
        """
        return sum(self.num_tokens_from_string(message) for message in messages)

    def num_tokens_from_string(self, string: str) -> int:
        """
        Count the number of tokens in the given `string`.
        """
        encoding = tiktoken.encoding_for_model(self.model.name)
        num_tokens = len(encoding.encode(str(string)))
        return num_tokens
