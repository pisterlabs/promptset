"""Building prompts objects."""
import html
import json
import os

from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from utils import format_dicts_to_string


class PromptBuilder:
    @classmethod
    def build_prompt_from_dir(cls, prompt_id: str) -> ChatPromptTemplate:
        """Builds a prompt given a prompt id.
        
        Prombt directory should contains:
        1- prombt.json: This file should contain prombt instructions and output format,
        ex:
        {
        "text": "prompt instructions",

        "output_format": "Prompt for the output format"
        }
        2- examples.json: in case using a few-shots prompting, this file should contain the 
        exmaples used in the few shots prompt as a list of jsons.
        Each json is for 1 example, and contains "text" key and "measurements" key.

        Args:
            prompt_id (str): Prompt Id, souhld be the directory name 
            where prompt files are. each example is a
            

        Returns:
            ChatPromptTemplate: CHat prompt.
        """

        prompt = cls._load_prompt_from_json(prompt_id)
        examples = cls._load_examples_from_json(prompt_id)

        return (
            cls.build_few_shots_prompt(prompt["text"], examples),
            prompt,
            examples,
        )

    @classmethod
    def build_few_shots_prompt(cls, prompt: str, examples: list) -> ChatPromptTemplate:
        """Given a propt instructions and examples, it build full few-shots prompt.

        Args:
            prompt (str): Prompt instructions.
            examples (list): List of dict holding the examples.

        Returns:
            ChatPromptTemplate: Full ChatPrompt object.
        """

        system_message = (
            "You are a helpful assistant you extract measurements from research patent."
        )
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_message
        )

        first_message = HumanMessagePromptTemplate.from_template(prompt)

        final_prompt = [system_message_prompt, first_message]

        for example in examples:
            final_prompt.append(
                HumanMessagePromptTemplate.from_template(
                    f"text: '''{example['text']}'''"
                )
            )
            final_prompt.append(
                AIMessagePromptTemplate.from_template(
                    format_dicts_to_string(example["measurements"])
                )
            )

        final_prompt.append(
            HumanMessagePromptTemplate.from_template(
                "\nOutput format:{output_format}:\n '''{input_text}'''"
            )
        )

        return ChatPromptTemplate.from_messages(final_prompt)

    @classmethod
    def _load_examples_from_json(cls, prompt_id: str) -> list[dict]:

        fie_path = f"Prompts/{prompt_id}/examples.json"
        if os.path.exists(fie_path):
            with open(fie_path, encoding="utf-8") as file:
                data = file.read()

            return json.loads(html.unescape(data))

        return []

    @classmethod
    def _load_prompt_from_json(cls, prompt_id):

        with open(f"Prompts/{prompt_id}/prompt.json", encoding="utf-8") as file:
            data = file.read()

        return json.loads(html.unescape(data))
