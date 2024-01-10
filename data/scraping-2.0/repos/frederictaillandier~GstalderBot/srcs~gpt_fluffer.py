""" Module to fluff text using GPT-3. """
from openai import OpenAI


class GPTFluffer:
    """Class to fluff text using GPT-3."""

    def __init__(self, key):
        self.client = OpenAI(api_key=key)

    def fluff(self, text) -> str:
        """Returns the fluffed text."""
        preprompt = (
            "You speak as the spirit protector of our house. Say in a poetic way : "
        )

        prompt: str = f"{preprompt} {text} \n\n"
        reponse = self.client.completions.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=1,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0,
        )
        return reponse.choices[0].text
