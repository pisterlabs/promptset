import os
import openai
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path

openai.api_key = os.getenv("OPENAI_API_KEY")


class ResultsCache:
    filename = Path("data", "interim", "openai_cache.json")

    def __init__(self):
        if self.filename.exists():
            with open(self.filename, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def __getitem__(self, key: str):
        return self.cache[key]

    def __setitem__(self, key: str, value: dict):
        self.cache[key] = value
        self.save()

    def save(self):
        with open(self.filename, "w") as f:
            json.dump(self.cache, f, indent=4)

    def stash_pairs(self, petitions: list[str], results: list[dict]):
        for petition, result in zip(petitions, results):
            self[petition] = result


def is_environmental_only_ten(
    list_of_petitions: list[str], recursion: int = 0
) -> list[dict[str, str]]:
    """
    Given a list of strings, will query the OpenAI API to determine if the petition is enviromental in nature.
    Only allows lists of length 1-10.
    """

    if openai.api_key is None:
        raise ValueError("Trying to use OpenAI, but OpenAI API key not found")

    if len(list_of_petitions) > 10:
        raise ValueError("List of petitions must be less than 10 items")

    base_prompt = """
    A list of petitions made to the UK Parliament, followed by an evaulation of if they cover environmental issues (true/false).
    
    Enviromental issues include themes or subjects such as:
    
    - climate change
    - net zero
    - carbon emissions
    - air pollution
    - water pollution
    - wildlife
    - ecology
    - forests
    - hunting
    - active travel
    - cycling
    - footpaths
    - coal mines
    - oil drilling
    - fracking

   Look for non-direct explanations, e.g. coal production can lead to higher carbon emissions. 

    The original list is a Json encoded list of strings.
    Return a json list as the input list with the structure ["result": bool, "explanation": str, "stub": str (first ten characters of petition)].
    JSON bools are true and false, not True and False.
    The input and the output must have the same length.

    Petition names:
    """

    prompt = base_prompt + json.dumps(list_of_petitions) + "\n\nOutput:\n"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256 * 4,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    try:
        data = json.loads(response.choices[0].text)  # type: ignore
    except json.JSONDecodeError:
        print(response.choices[0].text)  # type: ignore
        raise ValueError("OpenAI API returned invalid JSON")

    # if the output list is not the same length as the input list, try again
    if len(data) != len(list_of_petitions):
        if recursion > 5:
            print(data)
            raise ValueError("OpenAI API is not returning the full list of results")
        return is_environmental_only_ten(list_of_petitions, recursion=recursion + 1)

    return data


def is_environmental(
    list_of_petitions: list[str], ignore_cache: bool = False
) -> list[dict[str, str]]:
    """
    Given a list of strings, will query the OpenAI API to determine if the petition is enviromental in nature.
    """

    cache = ResultsCache()
    if ignore_cache:
        not_in_cache = list_of_petitions
    else:
        not_in_cache = [
            petition for petition in list_of_petitions if petition not in cache.cache
        ]

    if not_in_cache:
        # send 10 at a time
        results = []
        for i in tqdm(range(0, len(not_in_cache), 10)):
            results.extend(is_environmental_only_ten(not_in_cache[i : i + 10]))
        cache.stash_pairs(not_in_cache, results)

    return [cache[petition] for petition in list_of_petitions]
