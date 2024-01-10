import os
import openai

from fakes.FakeResponse import create_fake_response


class FakeOpenAiClient:
    def send_content(self, model, message, system_message):
        """
        Fake method to use instead of the openai client's method
        :param message: any message
        :return: response:obj, content:str
        """
        response = create_fake_response(message)
        return response, response.choices[0].message.content
