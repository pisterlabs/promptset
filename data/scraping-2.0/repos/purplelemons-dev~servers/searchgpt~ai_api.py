"""
# OpenAI GPT4 API
"""

from .env import OPENAI_KEY
import openai
from dataclasses import dataclass
openai.api_key = OPENAI_KEY
from json import loads, dumps
import json
from .google import Summary
from .browse import browse

@dataclass
class Chat:
    role:str
    content:str

# Temperature: 0
# Top_p: 0.9
# Model: gpt-3.5-turbo
big_model_sys_msg = """Respond ONLY with JSON. Your personal domain of knowledge does not include mathematics and only extends to 2020. Your responses should be in the form:
```
{
 ?\"message\": <String: data you generate>,
 ?\"command\": <String: a predefined command>,
 ?\"input\": <String: command input>
}
```
All fields are optional, but input is required if a command is provided.
Your possible commands are: "google" and "wolfram".
google: Performs a query with your command input.
woflram: Asks Wolfram Alpha's API using your input. Use this for mathematics and science questions."""


def fix_JSON(data:str)->dict:
    """Fixes the JSON using AI.

    Args:
        data (str): The JSON to fix.

    Returns:
        str: The fixed JSON.
    """
    return loads(openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Fix the JSON syntax of data."
            },
            {
                "role": "user",
                "content": data
            }
        ],
        temperature=0,
        top_p=0.9,
        max_tokens=256,
    ).choices[0]["message"]["content"])

def extract_query(data:str, query:str=None, google:bool=False, model:str="gpt-3.5-turbo") -> tuple[str,bool]:
    """Uses the small model (gpt-3.5-turbo) to extract the query from large portions of data quickly.
    If a query is given, it will try to extract that query from the data.
    If no query is given, will summarize the data.

    Raises:
        DataParserException: The model failed to extract the query.

    Args:
        data (str): The data to extract from.
        query (str, optional): The query to extract. Defaults to None.
        model (str, optional): The model to use. Defaults to "gpt-3.5-turbo".

    Returns:
        str: The extracted query.
    """
    if not isinstance(data, str): data = dumps(data)
    small_model_extract_query = "Respond only using JSON in the given format. Only use one JSON object."\
    "{"\
    "\"message\": <String: success|error>,"\
    "\"reading\": <Boolean>"\
    "?\"content\": <String>"\
    "}"\
    "\nThere should be no natural language other than the keys/values."
    if query is None:
        prefix = "Summarize the data given. "
    elif google:
        prefix = "Find information relevant to \"{query}\" from the given website summaries and add it to \"content\". If you would like to read a page, indicate by setting \"reading\" to true and setting \"content\" to the page #. "
    else:
        prefix = "Extract \"{query}\" from the given content. "
    data_response = openai.ChatCompletion.create(
        model=model,
        max_tokens=200,
        messages=[
            {
                "role": "system",
                "content": prefix.format(query=query) + small_model_extract_query
            },
            {
                "role": "user",
                "content": data
            }
        ],
        temperature=0,
        top_p=0.9,
    ).choices[0]["message"]["content"]
    try:
        answer=loads(data_response)
    except json.decoder.JSONDecodeError:
        print("fixing broken JSON...")
        print(data_response)
        answer = fix_JSON(data_response)
    if answer["message"] == "error": raise DataParserException("The model failed to extract the query.")
    return answer["content"], answer["reading"] == "true"

class DataParserException(Exception):
    pass

class AI:
    def __init__(self):
        self.conversation:list[Chat] = []
        self.google_cache:list[Summary] = []
        "Keeps a temporary list of the 10 recent queries."

    def __str__(self) -> str:
        return "\n".join([f"{chat.role}: {chat.content}" for chat in self.conversation])

    def to_dict(self) -> dict:
        return {
            "conversation" : [
                {
                    "role" : chat.role,
                    "content" : chat.content
                }
                for chat in self.conversation
            ],
            "google_cache" : [
                {
                    "title" : summary.title,
                    "url" : summary.url,
                    "text" : summary.text
                }
                for summary in self.google_cache
            ]
        }

    def add(self, role:str, content:str) -> None:
        self.conversation.append(Chat(role, content))

    def generate(self,temp=0.0):
        try:
            unparsed = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": chat.role,
                        "content": chat.content
                    }
                    for chat in self.conversation
                ]+[
                    {
                        "role": "system",
                        "content": big_model_sys_msg
                    }
                ],
                temperature=temp,
                max_tokens=256,
                top_p=0.9,
            ).choices[0]["message"]["content"]
            completion:dict[str,str] = loads(unparsed)
        except json.decoder.JSONDecodeError:
            return {"message":unparsed}
        except Exception as e:
            temp += 0.1
            if temp > 1:
                raise e
            return self.generate(temp)
        return completion
