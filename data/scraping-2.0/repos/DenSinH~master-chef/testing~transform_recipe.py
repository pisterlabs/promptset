import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

import os
import re
import json


openai.api_key = os.environ["OPENAI_API_KEY"]

MAX_RETRIES = 1
MODEL = "gpt-3.5-turbo"
PROMPT = """
The following text is from a website, and it contains a recipe, possibly in Dutch, as well as unnecessary other text from the webpage.
The recipe contains information on the ingredients, the prepration and possibly nutritional information.
Could you convert the recipe to a JSON object with the following keys:
"name": the name of this recipe.
"ingredients": a list of dictionaries, with keys "ingredient", mapping to the name of the ingredient, and "amount" which is a string containing the amount of this ingredient needed including the unit, or null if no specific amount is given.
               For example, the ingredient "one onion" should yield {{'amount': '1', 'ingredient': 'onion'}}, and the ingredient "zout" should yield {{'amount': null, 'ingredient': 'zout'}}.
"preparation": a list of strings containing the steps of the recipe.
"nutrition": null if there is no nutritional information in the recipe, or a list of dictionaries containing the keys "group", with the type
of nutrional information, and "amount": with the amount of this group that is contained in the recipe, as a string including the unit.
"url:": the literal string "{url}"
"people": the amount of people that can be fed from this meal as an integer, in case this information is present, otherwise null
"time": the time that this recipe takes to make in minutes as an integer, in case this information is present, otherwise null
"tags": interpret the recipe, and generate a list of at most 5 English strings that describe this recipe. For example, what the main ingredient is,
        if it takes long or short to make, whether it is especially high or low in certain nutritional groups, tags like that. Make
        sure the strings are in English.

Keep the language the same, except in the tags, and preferably do not change anything about the text in the recipe at all.
Only output the JSON object, and nothing else.
Here comes the text:

{text}
"""


class RecipeConversionError(Exception):
    pass


def translate_page(url):
    print("Retrieving URL")
    res = requests.get(url)
    if not res.ok:
        raise RecipeConversionError(f"Could not get the specified url, status code {res.status_code}")
    soup = BeautifulSoup(res.text, features="html.parser")

    # COMMENTS = ["comment", "opmerking"]
    # for attr in ["class", "id"]:
    #     for element in soup.find_all(attrs={attr: re.compile(fr".*({'|'.join(COMMENTS)}).*", flags=re.IGNORECASE)}):
    #         element.decompose()

    text = re.sub(r"(\n\s*)+", "\n", soup.text)
    prompt = PROMPT.format(url=url, text=text)

    print(f"Converting with ChatGPT ({MODEL})")
    messages = [
        {"role": "system", "content": "You are a helpful assistant that converts recipies into JSON format."},
        {"role": "user", "content": prompt}
    ]
    for i in range(1 + MAX_RETRIES):
        # todo: acreate
        chat_completion = openai.ChatCompletion.create(
            model=MODEL, messages=messages, temperature=0.2
        )
        reply = chat_completion.choices[0].message.content
        try:
            return json.loads(reply)
        except json.JSONDecodeError:
            print("Conversion failed, retrying")
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": "this is not a parseable json object, "
                                                        "only output the json object"})
    raise RecipeConversionError("ChatGPT did not return a parsable json object, please try again")


if __name__ == '__main__':
    from pprint import pprint

    recipe = translate_page("https://www.eefkooktzo.nl/wrap-mango-en-kip/")
    pprint(recipe)
    r"""
    Traceback (most recent call last):
      File "C:\Users\Dennis\PycharmProjects\masterchef\testing\transform_recipe.py", line 85, in <module>
        recipe = translate_page("https://www.eefkooktzo.nl/wrap-mango-en-kip/")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "C:\Users\Dennis\PycharmProjects\masterchef\testing\transform_recipe.py", line 68, in translate_page
        chat_completion = openai.ChatCompletion.create(
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "C:\Users\Dennis\PycharmProjects\masterchef\venv\Lib\site-packages\openai\api_resources\chat_completion.py", line 25, in create
        return super().create(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "C:\Users\Dennis\PycharmProjects\masterchef\venv\Lib\site-packages\openai\api_resources\abstract\engine_api_resource.py", line 153, in create
        response, _, api_key = requestor.request(
                               ^^^^^^^^^^^^^^^^^^
      File "C:\Users\Dennis\PycharmProjects\masterchef\venv\Lib\site-packages\openai\api_requestor.py", line 298, in request
        resp, got_stream = self._interpret_response(result, stream)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "C:\Users\Dennis\PycharmProjects\masterchef\venv\Lib\site-packages\openai\api_requestor.py", line 700, in _interpret_response
        self._interpret_response_line(
      File "C:\Users\Dennis\PycharmProjects\masterchef\venv\Lib\site-packages\openai\api_requestor.py", line 763, in _interpret_response_line
        raise self.handle_error_response(
    openai.error.InvalidRequestError: This model's maximum context length is 4097 tokens. However, your messages resulted in 4119 tokens. Please reduce the length of the messages.
    
    Process finished with exit code 1
    """
    # openai.error.InvalidRequestError: This model's maximum context length is 4097 tokens. However, your messages resulted in 4119 tokens. Please reduce the length of the messages.