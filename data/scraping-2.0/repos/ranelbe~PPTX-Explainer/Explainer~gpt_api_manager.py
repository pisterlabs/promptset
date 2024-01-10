import os
import openai
import backoff

# an environment variable
API_KEY = os.getenv("OPENAI_API_KEY")

class GptAPIManager:
    """ A class to manage the API calls to the OpenAI api. """

    def __init__(self):
        """
        Initialization.
        Set the API key from an environment variable.
        """
        openai.api_key = API_KEY

    @staticmethod
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    async def generate_answer(prompt: str) -> str:
        """
        Send a prompt to OpenAI API and get the answer.
        Use backoff mechanism to handle rate limit errors.
        :param prompt: The prompt to send.
        :return: The answer.
        """
        completion = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
