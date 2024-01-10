from __future__ import annotations

import os
import openai
import yaml

from abc import ABC

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

### ChatGPT interface to generate monty python scripts
class LLMAccessor(ABC):
    prompt_instructions = """ Write a novel Monty Python screenplay. """
    text_prompt = "text: What is your actual text answer to the prompt? Do not provide any other parameters. Only " \
                  "respond with your verbatim text answer to the prompt."

    def __init__(self):

        try:
            configs = yaml.safe_load(open("Parsers/LLM_config.yaml", "r"))
        except yaml.YAMLError:
            print("Error loading LLM configurations")
            exit(1)

        os.environ["OPENAI_API_KEY"] = configs["OPENAI_API_KEY"]
        openai.api_key = configs["OPENAI_API_KEY"]
        self._temperature = configs["temp"]
        self._model = configs["model"]
        self._prompt: str = "Give me a text file summarising some text."

    # temp getter
    @property
    def temperature(self) -> int:
        return self._temperature

    # temp setter
    @temperature.setter
    def temperature(self, temperature: int):
        self._temperature = temperature

    # model getter
    @property
    def model(self) -> str:
        return self._model

    # model setter
    @model.setter
    def model(self, m):
        self._model = m

    def generate_prompts(self, user_prompt) -> dict[str, str]:
        text_template = f"""
            Please give a full script for the following text:

            text: {user_prompt}

            Format the response as plain text. Each time a speaker says a line, the speaker's name should be in all 
            capitals and on a new line. The text a speaker says should be immediately below the speaker's name on a new
            line. This might look as follows:
            
                JOHN CLEESE:
                "Arthur, Arthur, King of the Britons, you have provided yourself worthy."

                LANCELOT:
                "Right. Who's with me?"
            
            The script should be 1970's tongue in cheek british humour, in the Monty Python style, based on the user input.
            """

        script_schema = ResponseSchema(
            name="script",
            description=""
        )

        response_schemas = [
            script_schema
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt_template = ChatPromptTemplate.from_template(template=text_template)

        messages = prompt_template.format_messages(
            format_instructions=format_instructions
        )

        chat = ChatOpenAI(temperature=self.temperature)

        response = chat(messages)
        print("*****************THIS IS THE RESPONSE***********************")
        print(response)

        return response


def create_script_with_ChatGPT(prompt):

    Accessor = LLMAccessor()
    return str(Accessor.generate_prompts(prompt))

