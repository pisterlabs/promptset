import json
import openai
from sentimeter.backend import Basebackend


class AIRemoteBackend(Basebackend):
    """
    Experimental ChatGPT backend
    """

    def __init__(self, OPENAI_KEY) -> None:
        super().__init__("ChatGPT")
        self.openai = openai
        self.openai.api_key = OPENAI_KEY
        # Set up the model and prompt
        self.model_engine = "text-davinci-003"
        self.prompt = """
        can you identify the levels of Happy, Sad , Fear, Angry and Surprise in the
         following test and share the result as json format?
        """

    def extract_valid_json(self, output):
        """Chatgpt sometimes returns non json text, lets strip it"""
        res = output[output.index("{") : output.index("}") + 1]
        return res

    def process(self, text):
        """Lets ask the badass !"""
        data = {}
        prompt_with_input = self.prompt + text
        completion = openai.Completion.create(
            engine=self.model_engine,
            prompt=prompt_with_input,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        # print("chat GPT Results")
        result = completion.choices[0].text
        result = result.replace("\n", " ")
        # print(result)
        json_string = self.extract_valid_json(result)
        try:
            data = json.loads(json_string)
        except ValueError:
            data = {}
        return data
