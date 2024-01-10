import os
import time

import openai
from openai.error import InvalidRequestError
from logger import generate_logger

logger = generate_logger(__name__)


def safe_chat_complete(max_attempts=10, chatcompletion_kwargs={}):
    interval = 5
    for attempt in range(1, max_attempts + 1):
        try:
            openai.api_key = os.environ['OPENAI_API_KEY']
            return openai.ChatCompletion.create(
                **chatcompletion_kwargs
            )
        except InvalidRequestError as e:
            if e.message.startswith('This model\'s maximum context length is 4097 tokens. However, your messages resulted in'):
                messages = chatcompletion_kwargs['messages']

                # remove one question and answer pair from first element
                # if success, will retry
                will_raise = True
                for i in range(len(messages)):
                    if i + 2 < len(messages):
                        if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                            messages = messages[:i] + messages[i+2:]
                            will_raise = False
                            break
                if will_raise:
                    raise e

            logger.warning(
                f'Attempt {attempt} failed with {e}, retrying in {interval} seconds', exc_info=True)
            time.sleep(interval)
            interval *= 2  # double the interval
        except Exception as e:
            logger.warning(
                f'Attempt {attempt} failed with {e}, retrying in {interval} seconds', exc_info=True)
            time.sleep(interval)
            interval *= 2  # double the interval
