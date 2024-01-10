import json

from openai import OpenAI

MODEL: str = "gpt-3.5-turbo"

EQ_PROMPT = """
Are these song JSON objects referring to more or less the same song? If the album is different please ignore that. Focus on the song and artist.

Please respond using the following format:

{
  "equivalent": <JSON Boolean>,
"confidence": <percentage string>
}
"""


class DataNormalization:
    """
    API wrapper for connecting to OpenAI's Data Normalization API
    """

    def __init__(self, openai_token: str):
        self.model = MODEL
        self.openai_token = openai_token
        self.client = OpenAI(api_key=openai_token)

    def health_check(self):
        return True, "OK"

    def check_equivalence(self, data1: str, data2: str):
        """
        Check if two strings are equivalent
        :param data1:
        :param data2:
        :return:
        """

        chat_completion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=100,
            messages=[
                {"role": "system", "content": EQ_PROMPT},
                {"role": "user", "content": f"{data1}\n\n{data2}"},
            ],
        )

        print(chat_completion.choices[0].message.content)

        return json.loads(chat_completion.choices[0].message.content)

    def normalize(self, prompt: str, data: str):
        """
        Normalize a given string of data
        :param data:
        :param prompt:
        :return:
        """

        if prompt is None:
            prompt = "Normalize the following data:\n\n" + data + "\n\nNormalized data:"

        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": data},
            ],
        )

        print(chat_completion.choices[0].message.content)

        return json.loads(chat_completion.choices[0].message.content)
