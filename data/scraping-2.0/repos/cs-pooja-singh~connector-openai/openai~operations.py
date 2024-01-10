import openai
from connectors.core.connector import get_logger, ConnectorError
from .constants import LOGGER_NAME
logger = get_logger(LOGGER_NAME)

def chat_completions(config, params):
    openai.api_key = config.get('apiKey')
    model = params.get('model', 'gpt-3.5-turbo')
    message = params.get('message')
    response = openai.ChatCompletion.create(model=model,
                                 messages=[
        {"role": "system", "content": "Be concise and helpful assistant."},
        {"role": "user", "content": message}
    ])
    return response

