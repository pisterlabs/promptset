import openai
import warnings

GPT_MODEL = "text-davinci-003"


class LightningGPT3:
    """This class component allows integrating GPT-3 into your Lightning App.
    The `generate()` method can be used to generate text from a prompt.
     This component just acts as a wrapper around the OpenAI API"""

    def __init__(self, api_key: str):
        super().__init__()

        openai.api_key = api_key
        try:
            openai.Model.list()
        except:
            raise Exception("Sorry, you provided an invalid API Key")

    def generate(self, prompt: str, max_tokens: int = 20):
        if max_tokens < 15:
            warnings.warn(
                "The value of the max_token variable is currently set ",
                max_tokens,
                ". To ensure that your prompts contain enough information, we recommend setting the max_token>=15.",
            )

        response = openai.Completion.create(model=GPT_MODEL, prompt=prompt, max_tokens=max_tokens, temperature=0.7)
        return response["choices"][0]["text"]
