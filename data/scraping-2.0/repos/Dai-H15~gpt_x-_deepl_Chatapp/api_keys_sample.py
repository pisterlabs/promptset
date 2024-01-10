"""Import libraries"""

import openai
import deepl


def set_apikey():
    openai.api_key = "any"
    openai.base_url = "any"
    translator = deepl.Translator("any")
    return translator
