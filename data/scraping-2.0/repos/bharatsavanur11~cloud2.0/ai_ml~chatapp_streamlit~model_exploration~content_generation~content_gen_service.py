from openai import OpenAI
from md_exp_configs.openai_config import get_openai_client
import os


def generate_content(prompt, model, temperature: 0.7, max_tokens: 400):
    '''
    Generates content based on prompt from vanilla open API Call
    :param model:
    :param temperature:
    :param max_tokens:
    :param prompt:
    :return:
    '''
    client = get_openai_client()
    completion = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return completion.choices[0].text

