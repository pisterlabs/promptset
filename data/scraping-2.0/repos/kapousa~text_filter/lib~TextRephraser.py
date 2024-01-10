import os
import re

import openai

from lib.constants.k import F, S, T


class TextRephraser:

    modes = ["Standard", "Fluency", "Formal", "Simple", "Creative"]
    text_process = {
        "rephrase": "Please rephrase the follwing text",
        "summary": "Please summary the follwing text"
    }

    def __init__(self):
        ''' Constructor for this class. '''
        self.k_path = ''

    def maintane_sentence(self, user_text, _process, _mode = "5"):
        oak = "{0}{1}{2}".format(F, S, T)
        openai.api_key = oak
        text_process = self.text_process[_process]
        text_mode = self.modes[int(_mode)]
        # Define the GPT-3 prompt that will be used to generate rephrased paragraphs
        text = "{0} in {1} way: {2}".format(text_process, text_mode, user_text)
        prompt = (text)
        # Define the OpenAI API parameters
        parameters = {
            "model": "text-davinci-002",
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

        # Call the OpenAI API to generate rephrased paragraphs
        response = openai.Completion.create(**parameters)

        # Extract the rephrased paragraphs from the OpenAI API response
        rephrased_paragraphs = response.choices[0].text

        # Clean up the rephrased paragraphs
        rephrased_paragraphs = re.sub('\n', '', rephrased_paragraphs)
        rephrased_paragraphs = re.sub('\t', '', rephrased_paragraphs)

        # Print the rephrased paragraphs
        return rephrased_paragraphs

    def reprashe_sentence(self, user_text):

        oak = "{0}{1}{2}".format(F, S, T)
        openai.api_key = oak
        # Define the GPT-3 prompt that will be used to generate rephrased paragraphs
        text = "Please rephrase the follwing text: {}".format(user_text)
        prompt = (text)
        # Define the OpenAI API parameters
        parameters = {
            "model": "text-davinci-002",
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

        # Call the OpenAI API to generate rephrased paragraphs
        response = openai.Completion.create(**parameters)

        # Extract the rephrased paragraphs from the OpenAI API response
        rephrased_paragraphs = response.choices[0].text

        # Clean up the rephrased paragraphs
        rephrased_paragraphs = re.sub('\n', '', rephrased_paragraphs)
        rephrased_paragraphs = re.sub('\t', '', rephrased_paragraphs)

        # Print the rephrased paragraphs
        return rephrased_paragraphs

    def summary_sentence(self, user_text):
        oak = "{0}{1}{2}".format(F, S, T)
        openai.api_key = oak
        # Define the GPT-3 prompt that will be used to generate rephrased paragraphs
        text = "Please summary the follwing text: {}".format(user_text)
        prompt = (text)
        # Define the OpenAI API parameters
        parameters = {
            "model": "text-davinci-002",
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

        # Call the OpenAI API to generate rephrased paragraphs
        response = openai.Completion.create(**parameters)

        # Extract the rephrased paragraphs from the OpenAI API response
        rephrased_paragraphs = response.choices[0].text

        # Clean up the rephrased paragraphs
        rephrased_paragraphs = re.sub('\n', '', rephrased_paragraphs)
        rephrased_paragraphs = re.sub('\t', '', rephrased_paragraphs)

        # Print the rephrased paragraphs
        return rephrased_paragraphs
