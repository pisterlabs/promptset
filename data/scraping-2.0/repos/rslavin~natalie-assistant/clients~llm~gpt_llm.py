import os

from clients.llm.llm_interface import LlmClient
from openai import OpenAI
from dotenv import load_dotenv
from timeout_function_decorator.timeout_decorator import timeout

MAX_RESPONSE_TOKENS = 200
MAX_CONTEXT_TOKENS = 2048
# MAX_CONTEXT_TOKENS = 448
MODEL = "gpt-4-1106-preview"
# MODEL = "gpt-3.5-turbo"

load_dotenv()


class GptLlm(LlmClient):

    def __init__(self, persona, max_response_tokens=MAX_RESPONSE_TOKENS, max_context_tokens=MAX_CONTEXT_TOKENS, model=MODEL):
        super().__init__(persona)
        self.max_response_tokens = max_response_tokens
        self.max_context_tokens = max_context_tokens
        self.model = model
        self.openai_client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

    @timeout(8)
    def response_generator(self, messages):
        """
        Converts openai chat completion generator chunks into text chunks.
        :param messages: List of messages of appropriate dictionaries
        :return:
        """

        messages = [{'content': d['content'], 'role': d['role'].__str__()} for d in messages]

        raw_generator = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.persona.temperature,
            max_tokens=MAX_RESPONSE_TOKENS,
            stream=True
        )

        for chunk in raw_generator:
            yield chunk.choices[0].delta.content

    # @timeout(15)
    # def get_response(self, message):
    #     self.append_message("user", message, to_disk=True)
    #     self.make_room()
    #
    #     chat = self.llm_client.chat.completions.create(
    #         model=MODEL,
    #         messages=self.get_conversation(),
    #         temperature=self.persona.temperature,
    #         max_tokens=MAX_RESPONSE_TOKENS
    #     )
    #     response = chat.choices[0].message.content
    #     self.append_message("assistant", response, to_disk=True)
    #
    #     return response
