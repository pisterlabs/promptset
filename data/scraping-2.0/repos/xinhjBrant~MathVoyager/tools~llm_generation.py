from key import *
from tenacity import (
    retry,
    wait_fixed,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import openai
import tiktoken
import re
import random

# start_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")
# os.makedirs('logs/main', exist_ok=True)
# logger = logging.getLogger(f'default')
# handler = logging.FileHandler(f"logs/main/{start_time}.log")
# formatter = logging.Formatter(
#     "%(asctime)s - %(message)s"
# )
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.setLevel(logging.INFO)

encoder = tiktoken.encoding_for_model("gpt-4")

available_models = [i['id'] for i in openai.Model.list(api_key=LLM_API_KEY, organization=ORG)['data']]

def model_random_choice(messages, model):
    token_count = sum(len(encoder.encode(m["content"])) for m in messages)
    model_choice = re.compile(model)
    if ('gpt-4' in model and token_count > 8190) or ('gpt-3.5' in model and token_count > 3000) or '32k' in model or '16k' in model:
        if models_choose := [i for i in available_models if model_choice.match(i) and '32k' in i or '16k' in i]:
            return random.choice(models_choose)
        else:
            return ''
    else:
        if models_choose := [i for i in available_models if model_choice.match(i) and '32k' not in i and '16k' not in i]:
            return random.choice(models_choose)
        else:
            return ''

@retry(wait=wait_random_exponential(min=1, max=16))
# @retry(wait=wait_fixed(1))
def llm_generate(logger, messages : list, model='gpt-4', batch_size=1, temperature=0):
    """
    The Python function `llm_generate(messages)` uses the OpenAI API to generate a chat completion. It uses the gpt-4 model and the messages provided as input. The function is decorated with a retry mechanism that waits for a random exponential time between 1 and 60 seconds before retrying, up to a maximum of 6 attempts.
    :param messages: a list of messages to be used for generating chat completion
    """
    model = model_random_choice(messages, model)
    if model == '':
        logger.info(f"Length exceed: {sum(len(encoder.encode(m['content'])) for m in messages)}")
        return ''
    # if '3.5' in model:
    #     key = random.choice(GPT_3_5_KEYS)
    # else:
    #     key = LLM_API_KEY
    key = LLM_API_KEY
    response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            api_key=key,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=batch_size,
            organization=ORG
        )
    for m in messages:
        logger.info(m["content"])
    logger.info(f"\033[33m{response['choices'][0]['message']['content']}\033[0m")
    logger.info(str({'prompt_tokens' : response['usage']['prompt_tokens'],
                 'completion_tokens' : response['usage']['completion_tokens'],
                 'total_tokens' : response['usage']['total_tokens']}))
    return response

@retry(wait=wait_random_exponential(min=1, max=16))
def llm_generate_with_function_calling(logger, messages : list, functions : list, model='.*'):
    model = model_random_choice(logger, messages, model)
    response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call="auto",  # auto is default, but we'll be explicit
            api_key=LLM_API_KEY,
            temperature=0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    logger.info(f"\033[33m{response['choices'][0]['message']['content']}\033[0m")
    return response