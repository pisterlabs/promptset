from . import prepare_pdf as prep
from . import variables
import openai
import sys
import os

openai.api_key = os.getenv('OPENAI_API_KEY')


def get_response(file_path: str):

    # Prepare the text for the API call
    messages = prep.format_to_messages(prep.text_cleaner(file_path))

    # OpenAI API call
    response = openai.ChatCompletion.create(
        model=variables.variable["model"],
        messages=messages,
        max_tokens=variables.variable["answer_size"],
        temperature=0,
    )["choices"][0]["message"]["content"]

    print(response)

    return response


if not os.path.isfile(sys.argv[1]):
    print(f"File path {sys.argv[1]} does not exist.")
    sys.exit(1)
else:
    get_response(sys.argv[1])
