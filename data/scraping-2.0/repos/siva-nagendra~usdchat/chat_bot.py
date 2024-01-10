import logging
import time
import traceback

import openai

from config.config import Config
from usdchat.error_handlers.openai_error_handler import handle_openai_error

openai.api_key = Config.OPENAI_API_KEY

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Chat:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config
        self.max_tokens = self.config.MAX_TOKENS
        self.temperature = self.config.TEMPERATURE

    def stream_chat(self, messages, delay_time=0.1):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages,
                stream=True,
            )
            reply_content = ""
            for event in response:
                event_text = event["choices"][0]["delta"]
                new_text = event_text.get("content", "")
                reply_content += new_text
                yield new_text
                time.sleep(delay_time)
            return reply_content

        except openai.error.APIError as e:
            error_message = handle_openai_error(401)
            logger.error(f"OpenAI API Error: {error_message}")
        except openai.error.APIConnectionError as e:
            logger.error("Failed to connect to OpenAI API.")
        except openai.error.RateLimitError as e:
            logger.error("OpenAI API request exceeded rate limit.")
        except openai.error.Timeout as e:
            logger.error("Request to OpenAI API timed out.")
        except Exception as e:
            logger.error(f"An exception occurred: {e}")
            traceback.print_exc()
