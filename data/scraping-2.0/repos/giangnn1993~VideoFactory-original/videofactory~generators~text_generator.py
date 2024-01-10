from .apis.llm.gpt4free_llm import gpt4freeLLM
from langchain import PromptTemplate, FewShotPromptTemplate
from typing import List
from ._prompts import examples_quote, prefix_quote, examples_image, prefix_image

import configparser
from pathlib import Path


class TextGenerator:
    LLM_CLASSES = {
        'g4f': gpt4freeLLM,
    }

    def __init__(self, llm_provider, processed_dir=None,
                 examples_quote=examples_quote, prefix_quote=prefix_quote,
                 examples_image=examples_image, prefix_image=prefix_image) -> None:
        self.llm_provider = llm_provider
        self.llm = self._create_llm_instance()
        self.processed_dir = processed_dir

        self.examples_quote = examples_quote  # Add examples_quote as a class variable
        self.prefix_quote = prefix_quote      # Add prefix_quote as a class variable
        self.examples_image = examples_image  # Add examples_image as a class variable
        self.prefix_image = prefix_image      # Add prefix_image as a class variable

        # Get the path of the current script (absolute path)
        current_script_path = Path(__file__).resolve()
        # Get the project folder (VideoFactory)
        project_folder = current_script_path.parent.parent.parent
        # Construct the path to config.ini in the parent folder
        config_path = project_folder / "config.ini"
        # Read the configuration file
        config = configparser.ConfigParser()
        config.read(config_path)

        # Use Path concatenation for processed_dir
        self.processed_dir = Path(processed_dir or
                                  project_folder / config.get('paths', 'processed_dir'))

    def _create_llm_instance(self):
        LLMClass = self.LLM_CLASSES.get(self.llm_provider)
        if LLMClass is None:
            raise ValueError(f'Unsupported LLM provider: {self.llm_provider}')
        return LLMClass()

    def set_llm_provider(self, llm_provider) -> None:
        self.llm_provider = llm_provider
        self.llm = self._create_llm_instance()

    def generate_chat_responses(self, query: str) -> List:
        return self.llm.generate_chat_responses(query)

    def create_few_shot_prompt_template(self, query: str, examples: str, prefix: str) -> str:
        # create an example template
        example_template = """
        User: {query}
        AI: {answer}
        """

        # create a prompt example from above template
        example_prompt = PromptTemplate(
            input_variables=["query", "answer"],
            template=example_template
        )

        # and the suffix our user input and output indicator
        suffix = """
        User: {query}
        AI: """

        # now create the few shot prompt template
        few_shot_prompt_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["query"],
            example_separator="\n"
        )
        return few_shot_prompt_template.format(query=query)


# USAGE
# ------------------------------------

# # Usage #1:
# # To use the TextGenerator class, create an instance:
# llm1 = TextGenerator('g4f')
# # Then, call the generate_chat_responses function with a query:
# query = "Tell a joke"
# responses = llm1.generate_chat_responses(query=query)
# # Iterate over the responses and print the response and provider name
# if responses is not None:
#     for response, provider_name in responses:
#         print("User:", query)
#         print(f'{provider_name}:', response)
# else:
#     print('Error occurred while generating chat response')


# # Usage #2:
# from _prompts import examples_quote, prefix_quote
# # To use the TextGenerator class to create an image prompt, create an instance:
# llm2 = TextGenerator('g4f')
# # Then, call the generate_chat_responses function with a query:
# query = "Family"
# prompt = llm2.create_few_shot_prompt_template(
#     query=query,
#     examples=examples_quote,
#     prefix=prefix_quote
# )
# responses = llm2.generate_chat_responses(query=prompt)
# # Iterate over the responses and print the response and provider name
# if responses is not None:
#     for response, provider_name in responses:
#         print("User:", query)
#         print(f'{provider_name}:', response)
# else:
#     print('Error occurred while generating chat response')


# # Usage #3:
# from _prompts import examples_image, prefix_image
# # To use the TextGenerator class to create an image prompt, create an instance:
# llm3 = TextGenerator('g4f')
# # Then, call the generate_chat_responses function with a query:
# query = "A human male merchant"
# prompt = llm3.create_few_shot_prompt_template(
#     query=query,
#     examples=examples_image,
#     prefix=prefix_image
# )
# responses = llm3.generate_chat_responses(query=prompt)
# # Iterate over the responses and print the response and provider name
# if responses is not None:
#     for response, provider_name in responses:
#         print("User:", query)
#         print(f'{provider_name}:', response)
# else:
#     print('Error occurred while generating chat response')
