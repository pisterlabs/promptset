# This example shows how to use the OpenAI API to classify the languages of a user input.
# see example input and out in example_input_sharegpt.json and example_output_lc.jsonl

import asyncio
import json
import logging
from typing import Any

import tiktoken
from instructor import OpenAISchema
from pydantic import BaseModel, Field

from openai_request_runner import process_api_requests_from_list


# Classes for OpenAI Functions
class LanguageCode(BaseModel):
    """ "A single language code in ISO 639-1 format"""

    lc: str = Field(..., description="Language code (e.g. 'en', 'de', 'fr')")


class LanguageClassification(OpenAISchema):
    """Classify the languages of a user prompt."""

    language_codes: list[LanguageCode] = Field(
        default_factory=list,
        description="A list of up to 2 languages present in the text. Exclude code sections, loanwords and technical terms in the text when deciding on the language codes. You have to output at least one language code, even if you are not certain or the text is very short!",
        max_items=2,
    )
    main_language_code: LanguageCode = Field(
        ..., description="Main Language of the text."
    )


# Functions for processing input and response


def preprocess_messages_sharegpt(request_json: dict, metadata: dict) -> list[dict]:
    # example for first 200 chars from user input in sharegpt4 format:

    assert request_json["items"][0]["from"] == "human"
    messages = [
        {
            "role": "system",
            "content": metadata["system_msg"],
        },
        {
            "role": "user",
            "content": request_json["items"][0]["value"][:200],
        },
    ]
    return messages


def postprocess_response(response, request_json: dict, metadata: dict) -> Any:
    """
    Postprocesses the API response to obtain language classification and related information.

    Args:
    - response (OpenAIObject): The response object from the OpenAI API call.
    - request_json (dict): The original request sent to the API.
    - metadata (dict): Metadata associated with the API request.

    Returns:
    - dict: A dictionary containing the language classification results and related information.
    """
    # customize for results
    try:
        lang_class = LanguageClassification.from_response(response)
    except AttributeError as e:
        logging.warning(
            f"Could not classify languages for {metadata['task_id']}, parsing error"
        )
        raise e

    encoding = tiktoken.get_encoding(metadata["token_encoding_name"])
    num_tokens = 0
    for message in request_json["items"]:
        try:
            num_tokens += len(encoding.encode(message["value"]))
        except Exception as e:
            logging.debug(
                f"Could not encode messages for {metadata['task_id']}, error {e}"
            )
            continue

    res_dict = {
        "num_languages": len(lang_class.language_codes),
        "main_language": lang_class.main_language_code.lc,
        "language_codes": [item.lc for item in lang_class.language_codes],
        "id": request_json["id"],
        "task_id": metadata["task_id"],
        "turns": len(request_json["items"]),
        "tokens": num_tokens,
    }
    return res_dict


# Load input data for processing
with open("examples/data/example_input_sharegpt.json", "r") as f:
    sharegpt_gpt4_train = json.load(f)


logging.basicConfig(level=logging.INFO)
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)
# Process the requests and obtain results
results = asyncio.run(
    process_api_requests_from_list(
        inputs=iter(sharegpt_gpt4_train),
        model="gpt-3.5-turbo",
        max_attempts=1,
        system_prompt="You are a world-class linguist and fluent in all major languages. Your job is to determine which languages are present in the user text and which one is the main language.",
        preprocess_function=preprocess_messages_sharegpt,
        postprocess_function=postprocess_response,
        functions=[LanguageClassification.openai_schema],
        function_call={"name": "LanguageClassification"},
        save_filepath="examples/data/example_output_lc.jsonl",
        raw_request_filepath="examples/data/example_raw_requests_lc.jsonl",
        check_finished_ids=False,
        num_max_requests=2,
        logging_level=20,
        debug=False,
    )
)
