import os
from typing import Tuple
import logging

import openai

from scenario.messages import MessageLog

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class AI:
    """ This is just a generic interface for an LLM API.
    It is separated from 'character_ai.py' to allow future expansion
    and clear boundaries between creation and use.
    Ideally, swap this interface out for LangChain or HuggingFace's transformers.
    """
    def __init__(self, model:str = "gpt-3.5-turbo"):
        openai.api_key = self._get_ai_key()
        self.model = model

    def _get_ai_key(self) -> str:
        key = os.environ.get("AI_KEY")
        if not key:
            try:
                with open('ai_key', "r") as txt_file:
                    key = txt_file.readlines()[0]
            except FileNotFoundError:
                logger.error("AI_KEY not found")
                raise Exception("AI_KEY not found")
        return key

    def ask(self, message_log:MessageLog, temperature:float = 0.0) -> Tuple[bool, str]:
        try:
            response = openai.ChatCompletion.create(
                model = self.model,
                messages =  message_log.messages,
                temperature = temperature, # explore vs exploit
            )
            logging.info(response)
            text_response = response.choices[0].message["content"]
        except openai.error.RateLimitError as E:
            return False, "I'm too tired to deal with you today. Someone will have to pay me to keep listening to you..."
        except openai.error.InvalidRequestError as E:
            return False, "I have no idea what you're even talking about."
        except AttributeError as E:
            return False, "Oh no! My moving skill has stopped working... Too bad for you!"
        except Exception as E:
            return False, "Oh no! My brain has stopped working... Too bad for you!"
        
        return True, text_response
