import requests
import json

from . import OPENAI_API_LINK, OPENAI_API_SECRET, SBER_API_LINK
from service import constants


def summarize(text: str, api="sber") -> str:
    """Summarizes the given text

    Args:
        text (str): the text to be summarized
        api (str, optional): Should be "openai" or "sber" depending on the api.
        Defaults to "sber".

    Returns:
        str: Summarized text
    """

    if api == "sber":
        return _make_sberapi_summarize_call(text)
    elif api == "openai":
        return _make_openai_summarize_call(text)
    else:
        raise ValueError("api arg must be 'openai' or 'sber'")


def _make_sberapi_summarize_call(text: str) -> str:
    """Makes a request to the basic sber api

    Args:
        text (str): the text to be summarized

    Returns:
        str: Summarized text
    """
    response = requests.post(SBER_API_LINK + "/predict", json={
        "instances": [
            {
                "text": text,
                "num_beams": 50,
                "num_return_sequences": 15,
                "length_penalty": 0.01
            }
        ]
    })

    response.raise_for_status()  # Raise an exception for any HTTP errors

    response_obj = json.loads(response.text)

    return response_obj['prediction_best']['bertscore']


def _make_openai_summarize_call(text: str,
                                prompt=""
                                       "Please provide a summary of all messages in the chat,"
                                       " organized by topics."
                                       " At first mention the topic,"
                                       " but put an appropriate emoji before you enter the topic title"
                                       " then in new line summarized opinions of participants,"
                                       " along with their names."
                                       " Write it in your own words and don't add minor details. DO NOT QUOTE."
                                       " Adhere to the given response format."
                                       " Answer in Russian. Here is the chat messages history in two possible formats: 1) 'username " + constants.SPLIT + " message text " + constants.SPLIT + "'"
                                                                                                                                                                             "All the text between symbols " + constants.SPLIT + " is a text of user's message."
                                       "2) 'username FORWARDED FROM forward_source_name " + constants.SPLIT + " message text " + constants.SPLIT + "'All the text between symbols " + constants.SPLIT + " is a text of the message which user forwarded from forward_source." ""
                                       "When making a sammari, keep in mind that this message is not the chat participant's own opinion, but is forwarded from another source."
                                                                                                                                                                                                                                 ":\n",
                                whole_summarize_prompt=""
                                                       "Please provide a summary based on the summaries below. You should aggregate them into"
                                                       "one whole summary. Follow their summary style: At first mention "
                                                       "the topic and put an appropriate emoji before you enter the topic title, then in new "
                                                       "line summarized opinions of participants along with their names. "
                                                       "Write it in your own words and don't add minor details. "
                                                       "You must answer in Russian.\n"
                                                       ""
                                ) -> str:
    """
    Makes a request to the 'gpt-3.5-turbo' version of chat gpt asking
    to summarize the provided text.

    Args:
        text (str): the text to be summarized
        prompt (str, optional): The prompt that will.
            Defaults to "Summarize the text below (try to be as short as possible but do not miss anything important):\n".

    Returns:
        str: Summarized text
    """

    model = "gpt-3.5-turbo"
    req_headers = {
        "Content-Type": "application/json",
        'Authorization': 'Bearer ' + OPENAI_API_SECRET
    }

    split_text = _split_string(text, max_size=4000-len(prompt))

    summarized_items = []

    for item in split_text:
        response = requests.post(OPENAI_API_LINK, headers=req_headers, json={
            "model": model,
            "temperature": 0.3,
            "messages": [{"role": "user", "content": prompt + " " + item}],
            "n": 1
        })

        response.raise_for_status()  # Raise an exception for any HTTP

        response_obj = json.loads(response.text)

        summarized_items.append(response_obj["choices"][0]["message"]["content"])

    whole_summarize_resp = requests.post(OPENAI_API_LINK, headers=req_headers, json={
        "model": model,
        "messages": [{"role": "user", "content": whole_summarize_prompt + " " + ' '.join(summarized_items)}],
        "n": 1
    })
    whole_summarize_resp.raise_for_status()
    whole_summarize_resp_obj = json.loads(whole_summarize_resp.text)
    return whole_summarize_resp_obj["choices"][0]["message"]["content"]


def _split_string(s, max_size=4000, delimiter='\n'):
    """
    Splits a string into chunks of a maximum size, using the specified delimiter.

    Args:
        s (str): The string to split.
        max_size (int, optional): The maximum size of each chunk. Defaults to 4000.
        delimiter (str, optional): The delimiter to use when splitting the string. Defaults to '\n'.

    Returns:
        List[str]: A list of strings representing the chunks of the original string.
    """
    lines = s.split(delimiter)
    result = []
    current_line = ''
    for line in lines:
        if len(current_line) + len(line) < max_size:
            current_line += line + delimiter
        else:
            result.append(current_line)
            current_line = line + delimiter
    if current_line:
        result.append(current_line)
    return result
