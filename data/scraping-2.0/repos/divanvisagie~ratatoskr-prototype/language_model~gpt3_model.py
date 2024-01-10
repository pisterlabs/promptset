import openai

from os import environ
from clients.openai_client import AI_STOP_TOKEN
from base_model import HUMAN_STOP_TOKEN, BaseModel
from log_factory.logger import create_logger

openai.api_key = TELEGRAM_BOT_TOKEN = environ['OPENAI_API_KEY']

text_model = "text-davinci-003"

logger = create_logger(__name__)

class GPT3CompletionModel (BaseModel):
    def __init__(self):
        pass
    
    def complete(self, prompt: str) -> str:
        try:
            logger.debug(f'Calling OpenAI {text_model} completion with Prompt:\n{prompt}')
            completion = openai.Completion.create(
                model=text_model,
                prompt=prompt,
                max_tokens=150,
                n=1,
                stop=[HUMAN_STOP_TOKEN, AI_STOP_TOKEN],
                temperature=0.9,
            )
            logger.debug(f'Returned: {completion}')
            message = completion.choices[0].text
            return message
        except Exception as e:
            logger.error(f'Failed to get answer from OpenAI: {e}')
            return 'A trickster seems to have disabled the Bifrost! We cannot reach the library of the gods.'