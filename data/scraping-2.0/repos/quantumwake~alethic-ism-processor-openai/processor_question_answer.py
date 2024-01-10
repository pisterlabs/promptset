from typing import List

import os.path
import openai
import dotenv
from core.base_processor import ThreadQueueManager, BaseProcessor
from core.processor_state import State
from core.processor_state_storage import ProcessorState, ProcessorStateStorage
from core.utils.general_utils import parse_response
from db.processor_state_db import BaseQuestionAnswerProcessorDatabaseStorage
from tenacity import retry, wait_exponential, wait_random, retry_if_not_exception_type
from logger import log
from openai import OpenAI

dotenv.load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY', None)
openai.api_key = openai_api_key


logging = log.getLogger(__name__)
logging.info(f'**** OPENAI API KEY (last 4 chars): {openai_api_key[-4:]} ****')


class OpenAIQuestionAnswerProcessor(BaseQuestionAnswerProcessorDatabaseStorage):

    def __init__(self,
                 state: State,
                 processor_state: ProcessorState,
                 processors: List[BaseProcessor] = None,
                 storage: ProcessorStateStorage = None,
                 *args, **kwargs):

        super().__init__(state=state,
                         storage=storage,
                         processor_state=processor_state,
                         processors=processors, **kwargs)

        self.manager = ThreadQueueManager(num_workers=10, processor=self)


    @retry(
        retry=retry_if_not_exception_type(SyntaxError),
        wait=wait_exponential(multiplier=1, min=4, max=10) + wait_random(0, 2))
    def _execute(self, user_prompt: str, system_prompt: str, values: dict):
        messages_dict = []

        if user_prompt:
            user_prompt = user_prompt.strip()
            messages_dict.append({
                "role": "user",
                "content": f"{user_prompt}"
            })

        if system_prompt:
            system_prompt = system_prompt.strip()
            messages_dict.append({
                "role": "system",
                "content": system_prompt
            })

        if not messages_dict:
            raise Exception(f'no prompts specified for values {values}')

        client = OpenAI()

        stream = client.chat.completions.create(
            model=self.model_name,
            messages=messages_dict,
            stream=False,
        )

        # final raw response, without stripping or splitting
        raw_response = stream.choices[0].message.content
        return parse_response(raw_response=raw_response)

