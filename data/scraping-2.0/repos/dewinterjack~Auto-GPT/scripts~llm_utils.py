import openai
from config import Config
cfg = Config()
from log_config import logger

openai.api_key = cfg.openai_api_key

# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=None, max_tokens=None)->str:
    if cfg.use_azure:
        response = openai.ChatCompletion.create(
            deployment_id=cfg.openai_deployment_id,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    

    logger.info(f"Number of tokens used: {response.usage.total_tokens}")
    logger.info(f"Number of prompt tokens used: {response.usage.prompt_tokens}")
    logger.info(f"Number of completion tokens used: {response.usage.completion_tokens}")
    logger.info(f"Request: {messages}")
    logger.info(f"Response: {response.choices[0].message['content']}")

    return response.choices[0].message["content"]
