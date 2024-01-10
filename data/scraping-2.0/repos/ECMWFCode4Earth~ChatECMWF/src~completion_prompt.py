"""
This module contains the main functions for leveraging the LLM capability to choose the rigth match among a list of different options. This operation is needed especially when using structured tools with a LLM. For example, let's say we want to integrate an API as a tool, employed by the LLM agent. The user inputs a query, the model translates this query into keyword arguments passed to the function implementation. The API needs however precisely the query parameters allowed from its schema; for example, for the image format, it may require a string amongst 'pdf', 'png', 'jpeg'. As for this example, with few choices it is easy, and can be implemented directly within the prompt introducing the tool. But for parameters with several tens of available items, such a strategy does not work, as we would require too many tokens for the prompt context. Therefore, this strategy explicitely asks for the proper API parameter. Within langchain, it should be possible to leverage pydantic data models for task, such that this would work out of the box with the [StructuredTool](https://blog.langchain.dev/structured-tools/) approach, but this is not yet working properly, as per this [issue](https://github.com/langchain-ai/langchain/issues/4724). 
"""
from typing import List

import openai

from .config import Logger, configs

openai.api_key = configs.OPENAI_API_KEY

prompt1 = """You have a list of available mappings. You have to extract from the list of mappings the one that best matches the description I will give you. Pick only from the available names. It may even be that there is not an exact match, but you need to find out which one is the best choice, always only from the available items. Please output only the name of the choice, no additional reasoning."""
prompt2 = """\n\nThe description is: '{}'"""
prompt3 = """\n\nThe list of available maps is: {}"""


def create_prompt(desc: str, maps: List[str]) -> str:
    """
    This generates the prompt starting from the base one and adding the input argument from the tool and the list of available parameters from the API schema.

    Args:
       desc: str
             the input argument from the tool
       maps: List[str]
             the available parameters from the API schema

    Returns:
       prompt: str
             the prompt to be passed for the completion
    """
    prompt = prompt1 + prompt2.format(desc) + prompt3.format(maps)
    return prompt


def get_chatgpt_completion(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    This function requests the answer for the prompt.

    Args:
        prompt: str
        model: str

    Returns:
        answer: str
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return str(response.choices[0].message["content"])


# Now we run a few examples
if __name__ == "__main__":
    maps = [
        "classical_global",
        "classical_europe",
        "classical_central_europe",
        "classical_north_west_europe",
        "classical_north_east_europe",
        "classical_south_west_europe",
        "classical_south_east_europe",
        "classical_northern_africa",
        "classical_north_atlantic",
        "classical_arctic",
        "classical_antarctic",
        "classical_north_america",
        "classical_central_america",
        "classical_south_america",
        "classical_eurasia",
        "classical_southern_asia",
        "classical_western_asia",
        "classical_eastern_asia",
        "classical_south_east_asia_and_indonesia",
        "classical_middle_east_and_india",
        "classical_southern_africa",
        "classical_australasia",
        "classical_west_tropic",
        "classical_east_tropic",
        "classical_equatorial_pacific",
        "classical_pacific",
        "classical_south_atlantic_and_indian_ocean",
        "classical_north_pole",
        "classical_south_pole",
    ]

    desc = "45N 12E"

    prompt = create_prompt(desc, maps)
    result = get_chatgpt_completion(prompt)
    print(f"{desc}: {result}")

    desc = "Venice"

    prompt = create_prompt(desc, maps)
    result = get_chatgpt_completion(prompt)
    print(f"{desc}: {result}")

    desc = "Venice near LA"

    prompt = create_prompt(desc, maps)
    result = get_chatgpt_completion(prompt)
    print(f"{desc}: {result}")

    desc = "Tokyo"

    prompt = create_prompt(desc, maps)
    result = get_chatgpt_completion(prompt)
    print(f"{desc}: {result}")

    desc = "Antarctica"

    prompt = create_prompt(desc, maps)
    result = get_chatgpt_completion(prompt)
    print(f"{desc}: {result}")
