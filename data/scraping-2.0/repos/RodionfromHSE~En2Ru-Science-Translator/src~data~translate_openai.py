from typing import List

import openai

default_prompt = """Translate the provided scientific texts from English to Russian. Before service, please make sure to clean up any messy or illegible symbols within the texts. Ensure that 
there are no additional explanations or content in the translations. The outputs must contain precisely the same 
information as the cleaned-up inputs."""


def translate_openai(string: str, system_prompt=default_prompt):
    """
    Translate a string from English to Russian using OpenAI's GPT-3 API
    :param string: String to translate
    :param system_prompt: The prompt to use for the system
    :return: A list of translated strings
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": string},
        ]
    )
    return completion.choices[0].message.content
