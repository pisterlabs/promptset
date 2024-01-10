import json
import openai
import os
from typing import Any, Dict, Tuple, Union
from dotenv import load_dotenv, find_dotenv

"""
I/O stuff 
"""

def read_dict_from_json(json_fname: str) -> Dict[str, Any]:
    with open(json_fname, "r") as f:
        return json.load(f)
    
def write_dict_to_json(json_fname: str, data_dict: Dict[str, Any]):
    with open(json_fname, "w") as f:
        json.dump(data_dict, f)

def read_json(json_fname: str) -> Any:
    with open(json_fname, "r") as f:
        return json.loads(json.load(f))

def write_json(json_fname: str, json_data: Any):
    with open(json_fname, "w") as f:
        json.dump(json.dumps(json_data), f)


def read_text(text_fname):
    with open(text_fname, "r") as f:
        return f.read()

"""
OpenAI stuff 
"""

def set_openai_api_key():
    """
    Provides openai api key from .env file to openai lib
    """
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ["OPENAI_API_KEY"]


def misc_get_completion(prompt: str, is_chat_model=True, is_return_total_tokens=False, stop=None) -> Union[str,Tuple[str, float]]:
    """
    Generates text given a prompt.
    is_chat_model (bool): uses GPT-3.5-TURBO if True, DAVINCI-002 otherwise.
    is_return_total_tokens (bool): get total tokens used if True, skipped otherwise
    """
    if is_chat_model:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= messages,
            temperature=0,
            stop=stop
        )
        completion = response.choices[0].message["content"]
    else:
        response = openai.Completion.create(
          model="text-davinci-002",
          prompt=prompt,
          temperature=0,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0.0,
          presence_penalty=0.0,
          stop=stop
        )
        completion = response["choices"][0]["text"]
    if is_return_total_tokens:
        return completion, response["usage"]["total_tokens"]
    else:
        return completion


def llm(prompt, stop=["\n"], model="text-davinci-002"):
    response = openai.Completion.create(
      model=model,# ,
      prompt=prompt,
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    return response["choices"][0]["text"]
