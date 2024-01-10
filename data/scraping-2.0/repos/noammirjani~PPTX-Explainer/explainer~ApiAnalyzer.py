"""

            ApiAnalyzer.py
    --------------------------------
 A class that analyze text by openai API
 Enter your API key in the API_KEY variable
"""
import backoff
import openai
import os
from constants import API_KEY

API_KEY = os.getenv("OPENAI_API_KEY")


class ApiAnalyzer:
    def __init__(self):
        self.check_api_key()
        self._api_key = API_KEY
        self.chat = self.set_connection_to_api()

    @staticmethod
    def check_api_key():
        if API_KEY == "YOUR API KEY":
            raise Exception("Please enter your API key in the API_KEY variable in ApiAnalyzer.py")

    def set_connection_to_api(self):
        """  set the connection to the API
        :return: the system prompt
        """
        openai.api_key = self._api_key
        system_prompt = "You're an AI text analyzer assisting with presentation summarization.For each slide's content"\
                        "you receive, generate a concise summary and additional explanation of the text, in case " \
                        "there is phrase you are not knowing try to understand from the complete text.\n"
        return [{"role": "system", "content": system_prompt}]

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    async def _get_explanation(self):
        """ analyze the text by request to the API, return the response
        :return: the response of the API
        """
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.chat, timeout=90)
        return completion.choices[0].message.content

    async def analyze(self, slide_content: str, index: int) -> dict:
        """ process the text by request to the API, return the response
         :param slide_content: the content of the slide
         :param index: the index of the slide
         :return: the response of the API
        """
        try:
            self._add_msg("user", slide_content)                 # set instructions to the chat as a user
            chat_response = await self._get_explanation()        # request and response
            self._add_msg("assistant", chat_response)            # keeping the history of the chat
            return {"slide_id": index, "analyze": chat_response + "\n"}
        except Exception as e:
            error_message = f"Error occurred while processing slide {index}: {str(e)}"
            return {"slide_id": index, "analyze": error_message}

    def _add_msg(self, role: str, content: str):
        """ add message to the chat
        :param role: the role of the message
        :param content: the content of the message
        """
        self.chat.append({"role": role, "content": content})
