"""GPTClient class to interact with OpenAI API"""

from os import path
from openai import OpenAI
from json_repair import loads
from jinja2 import Environment, FileSystemLoader
from ankicardmaker.languages import Language


class GPTClient:
    """Class providing a function to interact with OpenAI API"""

    def __init__(self):
        self.client = OpenAI()
        self.language = Language()
        self.prompt_template = self.get_prompt_template()

    def get_prompt_template(self):
        """Get prompt template."""
        template_path = path.join(path.dirname(__file__), "data")
        environment = Environment(
            loader=FileSystemLoader(template_path), autoescape=True
        )
        return environment.get_template("prompt.jinja")

    def create_prompt(self, text, language_code):
        """Create a prompt."""
        language_name = self.language.get_language_name(language_code)
        prompt = self.prompt_template.render(text=text, language_name=language_name)
        return prompt

    def get_gpt_response(self, prompt):
        """Get GPT response."""
        if not prompt:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")

        try:
            result = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as error:
            raise RuntimeError("Failed to get response from GPT API") from error

        if not result or not result.choices:
            raise ValueError("No response received from GPT API")

        gpt_response = str(result.choices[0].message.content)

        try:
            gpt_response = loads(gpt_response)
        except Exception as error:
            raise ValueError("Failed to parse response from GPT API") from error

        return gpt_response
