from datetime import datetime
import openai
import time
from src.logs import logger
from src.select_content.prompt_template import load_prompt_template
from .rate_limit import RateLimit
from src.apis.embedding import get_token_length
from .openai_api import chat_openai
from .azure_api import chat_azure


def retry_api(func):

    def wrapper(*args, **kwargs):
        retry_times = 3
        start_time = datetime.now()

        while retry_times > 0:

            try:
                answer = func(*args, **kwargs)
                retry_times = 0
            except openai.error.RateLimitError as e:
                logger.info('openai rate limit error, wait 90s.')
                logger.info(e)
                time.sleep(90)
                retry_times -= 1
            except openai.error.APIError as e:
                logger.info('openai server error')
                logger.info(e)
                time.sleep(90)
            except openai.error.ServiceUnavailableError as e:
                logger.info('openai server error')
                logger.info(e)
                time.sleep(90)
            except openai.error.Timeout as e:
                # TODO: repeat read new settings of sleep time
                logger.info('openai server error')
                logger.info(e)
                time.sleep(90)

        seconds = (datetime.now() - start_time).seconds
        answer['seconds'] = seconds
        return answer

    return wrapper


rate_limit = RateLimit()


@retry_api
def chat_ai(prompt, model="gpt-3.5-turbo", temperature=0):
    rate_limit = RateLimit()
    if rate_limit.chech_hit_context_length(prompt, model):
        logger.error(f'Prompt too long, {get_token_length(prompt)}')
        return {
            'answer': 'Prompt too long'
        }

    if rate_limit.check_hit_limit(prompt, model):
        logger.debug('Hit rate limit, wait 90s.')
        time.sleep(90)

    messages = [
        {"role": "system", "content": load_prompt_template('system')},
        {"role": "user", "content": prompt}]
    logger.debug(messages[0])

    # response = chat_openai(model, messages, temperature)
    response = chat_azure(model, messages, temperature)

    logger.debug(response)

    answer = response.choices[0].message.get("content")

    result = {
        'answer': answer,
        'completion_tokens': response.usage.completion_tokens,
        'prompt_tokens': response.usage.prompt_tokens,
        'total_tokens': response.usage.total_tokens,
    }

    return result
