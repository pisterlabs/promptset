import os
import json
from typing import List, Dict, Any

import openai

from logger import logger


class LLMInterface:
    def __init__(self):
        self.model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo-1106")

        openai.api_key = os.environ.get("OPENAI_API_KEY")
        if openai.api_key is None:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables")

    def send_messages(self, messages: List[Dict], function: Dict) -> Dict[str, Any]:
        logger.log_debug(f"Sending messages to OpenAI API: {messages}")
        try:
            resp = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                functions=[function],
                function_call={"name": function["name"]},
            )

            response_arguments = json.loads(resp["choices"][0]["message"]["function_call"]["arguments"])
            if not isinstance(response_arguments, dict):
                raise TypeError("The response from OpenAI API is not in the expected format.")
            return response_arguments

        except KeyError as ke:
            logger.log_error(f"KeyError during OpenAI API response parsing: {ke}")
            raise
        except ValueError as ve:
            logger.log_error(f"ValueError during OpenAI API response parsing: {ve}")
            raise
        except openai.error.OpenAIError as oe:
            logger.log_error(f"OpenAIError during API call: {oe}")
            raise
        except Exception as e:
            logger.log_error(f"An unexpected error occurred during trading instructions retrieval: {e}")
            raise
