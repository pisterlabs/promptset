import time

import openai
import os
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score

CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
api_key_file = f"{CURRENT_FILE_DIRECTORY}/../../../../data/open_ai_api_key.txt"

with open(api_key_file, "r") as apikey_file:
    api_key = apikey_file.readlines()[0].strip()
    if api_key == "ADD_YOUR_OPEN_AI_API_KEY_HERE":
        raise ValueError(
            f"You need to update Open AI API key in {api_key_file} in order to use GPT-3"
        )

openai.api_key = api_key


def postprocess_response(response: str) -> str:
    """
    Postprocess the response from GPT-3 to make it more human-readable.
    :param response:
    :return:
    """
    original = response
    response = response.strip()
    response = response.strip(".").strip()

    if response.lower() not in ["yes", "no", "not sure"]:
        if "." in response:
            response = response.split(".")[0]
        if "," in response:
            response = response.split(",")[0]

    print(f"{original}, {response}")
    return response.lower()


def classify_paper(
    paper: Dict[str, str], criteria: List[str], gpt_model="text-davinci-002"
) -> Tuple[List[str], str]:
    """
    Classify a paper using GPT-3, write a prompt for every criteria in the list for yes/no questions.
    :param gpt_model:
    :param paper:
    :param criteria:
    :return:
    """
    prompt = "You are conducting a literature review of neural retrieval models for domain specific tasks.\n"

    prompt += f"Title: {paper.get('title')}\n"
    prompt += f"Abstract: {paper.get('abstract')}\n"
    prompt += f"Authors: {paper.get('authors')}\n"
    prompt += f"Journal: {paper.get('venue')}\n"
    prompt += f"Year: {paper.get('year')}\n"

    responses = []
    for criterion in criteria:
        prompt_template = f"{prompt}\n\nDoes the paper {criterion}? (yes/no/not sure)"

        try:
            response = openai.Completion.create(
                engine=gpt_model,
                prompt=prompt_template,
                temperature=0.9,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
            )
        except openai.error.RateLimitError:
            time.sleep(62)
            response = openai.Completion.create(
                engine=gpt_model,
                prompt=prompt_template,
                temperature=0.9,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
            )

        try:
            output = postprocess_response(response=response["choices"][0]["text"])
            responses.append(output)
        except IndexError:
            responses.append("not sure")

    return responses
