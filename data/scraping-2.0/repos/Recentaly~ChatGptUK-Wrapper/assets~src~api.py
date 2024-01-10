# ----------------------------------- IMPORTS ----------------------------------- #

# requests module to send http requests
import requests

# fake_useragent module to generate random user agents
from fake_useragent import UserAgent

# json module to parse json data
import json

# module to get headers
from .headers.get_headers import get_headers

# typing module for type hinting
from .typing import Headers, Messages, Any, Dict

# secrets module for random numbers
from secrets import randbelow

# logging module to log errors
import logging

# ----------------------------------- LOGGING CONFIG ----------------------------------- #

# get logger
logger: logging.Logger = logging.getLogger(__name__)

# set logging level
logger.setLevel(logging.DEBUG)

# basic config
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S"
)

# ----------------------------------- API CLASS ----------------------------------- #

class API(object):

    # constructor
    def __init__(self) -> None:

        """API class constructor"""

        # url to send http requests to
        self.url: str = "https://free.chatgpt.org.uk/api/langchain/tool/agent/nodejs"

        # get headers
        self.headers: Headers | Dict[str, str] = get_headers(UserAgent().random, randbelow(10000))

        # log
        logger.info(f"Headers successfully generated.\n")

        # pre-made config for data
        self.base_url: str = "/api/openai"
        self.use_tools: list[str] = ["web-search", "calculator", "web-browser"]
        self.max_iterations: int = 10
        self.use_intermediate_steps: bool = True
        self.azure_api_version = "2023-08-01-preview"
        self.is_azure: bool = False

        # log 
        logger.info(f"Config successfully generated.\n")

        # pre-make a session
        self.session: requests.Session = requests.Session()
    
    # this method retrieves a list of available models
    def get_models(self) -> Dict[str, Dict[str, str]]:

        """Get model used"""

        # logging info
        logger.info(f"Fetching models...\n")

        return {"data": [
            {"id": "gpt-3.5-turbo"},
            {"id": "gpt-3.5-turbo-0301"},
            {"id": "gpt-3.5-turbo-1106"},
            {"id": "gpt-3.5-turbo-0613"},
            {"id": "gpt-3.5-turbo-16k"},
            {"id": "gpt-3.5-turbo-16k-0613"},
            {"id": "gpt-4"},
            {"id": "gpt-4-32k"},
            {"id": "gpt-4-1106-preview"},
            {"id": "claude-instant-1.2"},
            {"id": "claude-2.0"},
            {"id": "claude-2.1"},
        ]}

    # this method allows you to chat with chatgpt
    def chat(self,
            messages: Messages,
            model: str,
            temperature: int = 0.6,
            presence_penalty: int = 0,
            frequency_penalty: int = 0,
            top_p: int = 1,
        ) -> str:

        """Chat with ChatGPT"""

        # logging info
        logger.info(f"Chat requested...\n")

        # data to send
        data: Dict[str, Any] = {
            "azureApiVersion": self.azure_api_version,
            "baseUrl": f"{self.base_url}",
            "frequency_penalty": frequency_penalty,
            "isAzure": self.is_azure,
            "maxIterations": self.max_iterations,
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            "top_p": top_p,
            "returnIntermediateSteps": self.use_intermediate_steps,
            "useTools": self.use_tools,
            "stream": True
        }

        # if model is from anthropic, add a maxTokens key and filter out the suffix
        if "claude" in model:

            data["max_tokens"] = None
            data.pop("azureApiVersion", None)
            data.pop("isAzure", None)
            data.pop("useTools", None)
            data.pop("baseUrl", None)
            data.pop("returnIntermediateSteps", None)
            data.pop("maxIterations", None)

            # delete first two messages of list (user message MUST be first)
            for message in data["messages"][:2]:

                if data["messages"][0]["role"] == "assistant" or "system":

                    data["messages"].pop(data["messages"].index(message)) 


        # logging info
        logger.info(f"Data successfully generated.\n")

        # send http request with session
        with self.session.post(self.url, headers=self.headers, json=data) as response:

            # logging info
            logger.info(f"Http request successfully sent.\n")

            # raise exception if http request failed
            response.raise_for_status()

            # logging info
            logger.info(f"Http request successfully received.\n")

            # iterate over response
            for line in response.iter_lines():

                # check if line is not empty
                if line:

                    # parse json data
                    delta_chunk: Dict[str, Any] = json.loads(line.decode('utf-8').removeprefix("data: "))

                    # filter out the "input" key. When web search is used, the "input" key is present in the response and it looks.. ugly.. so we filter it out
                    if '{"input":' not in delta_chunk["message"]:

                        # yield message
                        yield delta_chunk["message"]

        # regenerate session
        self.session = requests.Session()
