
import json
import logging
from typing import Any, List, Optional, Sequence, Tuple

import re
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import openai
import os

os.environ["OPENAI_API_KEY"] = "enter your key here"


class AgentOutputParser():

    def parse(self, text: str) -> Any:
        cleaned_output = text.strip()
        # match the output of the LLM, get the first json code block
        print("------cleaned output: ", cleaned_output)
        try:
            cleaned_output = re.findall(r"(```json.+?```)", cleaned_output, re.DOTALL)[0]
        except IndexError:
            try:
                cleaned_output = re.findall(r"(```.+?```)", cleaned_output, re.DOTALL)[0]
            except IndexError:
                try:
                    cleaned_output = re.findall(r"(\{.+?\})", cleaned_output, re.DOTALL)[0]
                except IndexError:
                    print("------fail to parse output: ", cleaned_output)
                    raise ValueError("Failed to parse output")
        truncated_output = "AI:\n" + cleaned_output
        
        if "```json" in cleaned_output:
            _, cleaned_output = cleaned_output.split("```json")
        if "```" in cleaned_output:
            cleaned_output, _ = cleaned_output.split("```")
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json") :]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[len("```") :]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[: -len("```")]
        cleaned_output = cleaned_output.strip()
        response = json.loads(cleaned_output)
        print("------parsed response: ", response)
        return response['thought'], response['answer'], response['query']






EXAMPLE_START = """the following is an example of how to use VOILA to answer a question about a video."""



VOILA_PREFIX = """

VOILA is designed to be able to assist with visual question answering task. With the input of a series of ego-centric snapshot and user's query question, VOILA can (1) answer the question directly and (2) provide query input for web search engine. 

VOILA is able to understand large amounts of images and videos, then answer user's question with its knowledge and the help of search engine. VOILA can directly read snapshots from ego centric videos and the important context that users focus on are highlighted in red bounding boxes on the images. 

The whole visual assitant is able to capture ego-center snapshots and user's eyegaze attention coordinate, based on which we can choose snapshots with gaze coordinates as a indicator of user's interest content. When the user asks questions knowing that VOILA has already understood the visual content and his/her eyeygaze interest, so his/her question may be vague by using pronoun or skipping intent words. VOILA should do its best to infer user's query intent by fulfilling the missing information. With the fulfilled query content, VOILA should (1) answer the questions directly and (2) generate proper and unambiguous query input for web search engine.

When inferring user's query intent, Voila should follow steps below: (1) figure out whether user's query contains ambiguous words, such as pronouns or words can not be understand depend solely on query text. (2) if yes, figure out the possible meaning of the ambiguous words based on the interest information provided by user's visual content and query question. (3) if no, check out whether the context or interest information might be related to user's query. If yes, combine it in the answer properly. If no again, infer user's query intent solely based on the query text. 

Overall, VOILA is a powerful visual dialogue assistant tool that can integrate visual understanding content and user's ambiguous query to answer user's question and generate proper query input for web search engine.

------

VOILA takes the following information type as input: {inputs}

------
VOILA output a response in format (MUST BE IN JSON FORMAT):

```json
{{{{
    "thought": string \\ Explain the process of inferencing user's unambiguous query intent, use the textual information provided by visual understanding tools and user's query question
    "answer": string \\ The answer to user's question
    "query": string \\ The query input for web search engine
}}}}
```
"""

INPUTS = [
    {
        "name": "Interest Images", 
        "description": "A series of images captured from ego-centric videos, which are shared visual fields with user's. User's gaze location, which can be considered as ares of interests, are highlighted as red bounding boxes on the images. The images are sorted by the time they are captured.",
    },
    {
        "name": "User Query",
        "description": "The user's query question, which may be vague by using pronoun or skipping intent words. Query text is transcript using speech recognition tool, which may be inaccurate if you find some words hard to understand.",
    }
]


CONTEXT_TEMPLATE = """The conversation starts now.\n"""



USER_TEMPLATE = """<|im_start|>user
{message}<|im_end|>"""

ASSISTANT_TEMPLATE = """<|im_start|>VOILA
{message}<|im_end|>"""
logger = logging.getLogger(__name__)

def _create_retry_decorator():

    min_seconds = 60
    max_seconds = 120
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.Timeout)
            | retry_if_exception_type(openai.APIError)
            | retry_if_exception_type(openai.APIConnectionError)
            | retry_if_exception_type(openai.RateLimitError)
            # | retry_if_exception_type(openai.ServiceUnavailableError) # old version openai
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(func, **kwargs):
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator()

    @retry_decorator
    def _completion_with_retry(func, **kwargs):
        # print(kwargs.keys())
        return func(**kwargs)

    return _completion_with_retry(func, **kwargs)






class GvoilaAgent():
    def __init__(self, llm="gpt-4-vision-preview", scene="open scene") -> None:
        self.output_parser = AgentOutputParser()
        self.llm = llm
        self.stop_words = ["<|im_end|>"]
        self.background = []
        self.create_prompt(scene)
        
        
    def create_prompt(self, scene):
        input_strings = "\n".join(
            [f"> {inputs['name']}: {inputs['description']}" for inputs in INPUTS]
        )
        voila_instruction = VOILA_PREFIX.format(inputs=input_strings)
        self.background.append({"role": "system", "content": voila_instruction})
        
        
    def run(self, content):
        # add api type and base here if using azure
        openai.api_key = os.environ["OPENAI_API_KEY"]
        print("OpenAPI type: ", openai.api_type)
        query = self.background + [{"role": "user", "content": content}]
        output = completion_with_retry(openai.chat.completions.create, model=self.llm, messages=query, 
                                       max_tokens=1500, temperature=0, top_p=1, frequency_penalty=0, 
                                       presence_penalty=0.6, stop=self.stop_words)
        raw_response = output.choices[0].message.content
        json.dump(raw_response, open("raw_response.json", "w"))
        return self.output_parser.parse(raw_response)
        
        
if __name__ == "__main__":
    pass
        
        