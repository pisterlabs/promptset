from __future__ import annotations

import api
# See https://platform.openai.com/docs/libraries/python-library
# https://platform.openai.com/docs/api-reference/chat/create
# Also https://pypi.org/project/openai/
import openai
from textwrap import dedent
# https://regexr.com/
import re

from typing import TypeVar, Iterable, NamedTuple, List, Tuple, Dict

openai.api_key = api.get_openapi_api_key()

MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 100


def send_prompts(prompts: Iterable[str] = None, log=False) -> List[str]:
    """
    Sends a series of prompts to ChatGPT and receives responses.

    :param prompts: the series of prompts.
    :param log: print the response.
    :return: a list of ChatGPT's responses corresponding to each prompt.
    """
    if prompts is None:
        prompts = ["Hello!"]
    response = openai.ChatCompletion.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt} for prompt in prompts
        ],
    )
    if log:
        print(f"Received raw response: {response}")
    return [choice["message"]["content"] for choice in response.choices]


Place = TypeVar("Place")


class TopPlaceQuery(NamedTuple):
    keyword: str
    places: Dict[Place, str]


def create_choices_prompt(places: Dict[Place, str]) -> Tuple[str, Dict[str, Place]]:
    choices = ""
    choice_to_place = {}

    for i, (place, description) in enumerate(places.items()):
        identifier = f"[{i + 1}]"
        choice_to_place[identifier] = place
        choices += f"{identifier}: {description}\n"

    return choices, choice_to_place


def get_top_place(query: TopPlaceQuery, log=False) -> Place | None:
    """
    Given a query consisting of a desired quality and a mapping of places to descriptions,
    ask ChatGPT to choose the best place for the quality.
    Results in a place or None if the prompt to ChatGPT fails or is in an unrecognizable format.

    :param query: the list of places.
    :param log: print the prompts and responses for debugging.
    :return: the place that best matches the desired quality or None if the query failed.
    """
    # prompt engineering
    prologue = dedent("""
        I'll give you a desired quality and a list of descriptions. 
        Select the description that is the best example of the desired quality. 
        Respond with the number of the description, such as [1].
        
        Here is one example:
        The desired quality is rustic.
        [1]: A high school classroom within a bustling city.
        [2]: A modern bedroom within a suburb.
        [3]: A quaint barn house in Iowa.
        [4]: A coral atoll in the middle of the Pacific Ocean.
        The answer is [3].
    """)

    desired_quality = query.keyword

    quality_desc = dedent(f"""
        The desired quality is {desired_quality}.
    """)

    choices, choice_to_place = create_choices_prompt(query.places)

    epilogue = "The answer is "

    prompt = prologue + quality_desc + choices + epilogue

    if log:
        print(f"Created prompt:\n{prompt}")

    responses = send_prompts([prompt], log=log)
    response = responses[0]

    if log:
        print(f"Received response: {response}")

    # try to parse the response

    # Regex for `[3]`, `[92130]`, etc.
    matches = re.findall("\[[0-9]+\]", response)
    if len(matches) > 0:
        try:
            return choice_to_place[matches[0]]
        except KeyError:
            if log:
                print(f"Response {response} did not match input choices!")

    print(f"Response {response} did not match known formats!")
    return None


def rank_places(query: TopPlaceQuery, log=False) -> list[Place] | None:
    """
    Given a query consisting of a desired quality and a mapping of places to descriptions,
    ask ChatGPT to rank the places for the quality.
    Results in the ranked list of places or None if the prompt to ChatGPT fails or is in an unrecognizable format.

    :param query: the query.
    :param log: print the prompts and responses for debugging.
    :return: the places ranked by desired quality or None if the query failed.
    """
    # prompt engineering
    prologue = dedent("""
        I'll give you a desired quality and a list of descriptions. 
        Rank the descriptions by how well they match the desired quality. 
        Respond with the numbers of the description, such as [1], [2], [3].

        Here is one example:
        The desired quality is rustic.
        [1]: A high school classroom within a bustling city.
        [2]: A modern bedroom within a suburb.
        [3]: A quaint barn house in Iowa.
        The answer is [3], [2], [1].
    """)

    desired_quality = query.keyword

    quality_desc = dedent(f"""
        The desired quality is {desired_quality}.
    """)

    choices, choice_to_place = create_choices_prompt(query.places)

    epilogue = "The answer is "

    prompt = prologue + quality_desc + choices + epilogue

    if log:
        print(f"Created prompt:\n{prompt}")

    responses = send_prompts([prompt], log=log)
    response = responses[0]

    if log:
        print(f"Received response: {response}")

    # try to parse the response

    # Regex for `[3]`, `[92130]`, etc.
    matches = re.findall("\[[0-9]+\]", response)
    if len(matches) > 0:
        try:
            return list(map(lambda match: choice_to_place[match], matches))
        except KeyError:
            if log:
                print(f"Response {response} did not match input choices!")

    if log:
        print(f"Response {response} did not match known formats!")
    return None


if __name__ == '__main__':
    # print(send_prompts(["Give me an example of a 'warm' location."], log=True))
    query1 = TopPlaceQuery(
        keyword="warm",
        places={
            "NYC":
                "The largest city in the United States.",
            "Central Park":
                "Central Park is an urban park in New York City. It is the fifth-largest park in the city.",
            "Los Angeles Library":
                "The Los Angeles Public Library provides free and easy access to information, ideas, books and technology that enrich, educate and empower every individual in our city's diverse communities."
        }
    )
    query2 = TopPlaceQuery(
        keyword="scenic",
        places={
            "Yosemite National Park":
                dedent(
                    """
                    Yosemite National Park is in California’s Sierra Nevada mountains. 
                    It’s famed for its giant, ancient sequoia trees, and for Tunnel View, 
                    the iconic vista of towering Bridalveil Fall and the granite cliffs of El Capitan and Half Dome.
                    """
                ),
            "Kresge Auditorium":
                dedent(
                    """
                    Kresge Auditorium is an auditorium structure at the Massachusetts Institute of Technology, 
                    located at 48 Massachusetts Avenue, Cambridge, Massachusetts.
                    """
                )
        }
    )
    print(rank_places(query1, log=True))
    # prologue = dedent("""
    #         I'll give you a desired quality and a list of descriptions.
    #         Rank the descriptions by how well they match the desired quality.
    #         Respond with the numbers of the description, such as [1], [2], [3].
    #
    #         Here is one example:
    #         The desired quality is rustic.
    #         [1]: A high school classroom within a bustling city.
    #         [2]: A modern bedroom within a suburb.
    #         [3]: A quaint barn house in Iowa.
    #         The answer is [3], [2], [1].
    #     """)
    # print(send_prompts([prologue], log=True))
