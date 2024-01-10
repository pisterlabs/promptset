import random
import time
from typing import Any, Dict, List

from spacy.tokens import Doc

from conspiracies.docprocessing.relationextraction.gptprompting import PromptTemplate
from conspiracies.registry import registry


@registry.prompt_apis.register("conspiracies/openai_gpt3_api")
def create_openai_gpt3_prompt_api(
    prompt_template: PromptTemplate,
    api_key: str,
    model_name: str,
    api_kwargs: Dict[Any, Any],
):
    def openai_prompt(targets: List[str]) -> List[str]:
        """"""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The OpenAI API requires the openai package to be installed. "
                "You can install the requirements for this module using "
                "`pip install conspiracies[openai]`.",
            )

        openai.api_key = api_key

        responses: List[str] = []
        for target in targets:
            # Run loop until reaching return and response is returned
            while True:
                try:
                    response = openai.Completion.create(
                        model=model_name,
                        prompt=prompt_template.create_prompt(target),
                        **api_kwargs,
                    )
                    responses.append(response["choices"][0]["text"])
                    break

                except openai.error.InvalidRequestError as e:
                    print("Invalid request got error: ", e)
                    print("Retrying with fewer examples...")
                    # Randomly select an example to drop
                    current_examples: List[Doc] = list(prompt_template.examples)
                    current_examples.pop(random.randrange(len(current_examples)))
                    prompt_template.set_examples(current_examples)  # type: ignore
                except openai.error.APIConnectionError:
                    print("Connection reset, waiting 20 sec then retrying...")
                    time.sleep(20)

        return responses

    return openai_prompt


@registry.prompt_apis.register("conspiracies/openai_chatgpt_api")
def create_openai_chatgpt_prompt_api(
    prompt_template: PromptTemplate,
    api_key: str,
    model_name: str,
    api_kwargs: Dict[Any, Any],
):
    def openai_prompt(targets: List[str]) -> List[str]:
        """"""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The OpenAI API requires the openai package to be installed. "
                "You can install the requirements for this module using "
                "`pip install conspiracies[openai]`.",
            )

        openai.api_key = api_key
        message_example = prompt_template.create_prompt("test")
        assert isinstance(message_example, list) and isinstance(
            message_example[0],
            dict,
        ), "ChatGPT requires a list of message dicts. Consider using chatGPTPromptTemplate as template."  # noqa: E501

        responses: List[str] = []
        for target in targets:
            # Run loop until reaching return and response is returned
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model=model_name,
                        messages=prompt_template.create_prompt(target),
                        **api_kwargs,
                    )
                    responses.append(response["choices"][0]["message"]["content"])
                    break

                except openai.error.InvalidRequestError:
                    # Randomly select an example to drop
                    current_examples = prompt_template.examples
                    current_examples.pop(random.randrange(len(current_examples)))
                    examples = current_examples
                    prompt_template.set_examples(examples)  # type: ignore

                except openai.error.ConnectionResetError:
                    print("Connection reset, waiting 20 sec then retrying...")
                    time.sleep(20)

        return responses

    return openai_prompt
