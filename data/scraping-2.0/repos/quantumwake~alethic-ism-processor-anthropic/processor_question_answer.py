import os
from typing import List

import dotenv
from logger import log

from anthropic import HUMAN_PROMPT, AI_PROMPT, Anthropic
from core.base_processor import ThreadQueueManager, BaseProcessor
from core.processor_state import State
from core.processor_state_storage import ProcessorState, ProcessorStateStorage
from core.utils.general_utils import parse_response_strip_assistant_message
from db.processor_state_db import BaseQuestionAnswerProcessorDatabaseStorage
from tenacity import retry, wait_exponential, wait_random, retry_if_not_exception_type

dotenv.load_dotenv()

logging = log.getLogger(__name__)
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", None)
logging.info(f'**** ANTHROPIC API KEY (last 4 chars): {anthropic_api_key[-4:]} ****')


class AnthropicQuestionAnswerProcessor(BaseQuestionAnswerProcessorDatabaseStorage):

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

        self.manager = ThreadQueueManager(num_workers=3, processor=self)

    @retry(retry=retry_if_not_exception_type(SyntaxError),
           wait=wait_exponential(multiplier=1, min=4, max=10) + wait_random(0, 2))
    def _execute(self, user_prompt: str, system_prompt: str, values: dict):
        anthropic = Anthropic(max_retries=5)

        final_prompt = f"{HUMAN_PROMPT} {user_prompt} {AI_PROMPT}"
        if system_prompt:
            final_prompt = f'{system_prompt} {final_prompt}'

        # strip out any white spaces and execute the final prompt
        final_prompt = final_prompt.strip()
        completion = anthropic.completions.create(
            model=self.config.model_name,
            max_tokens_to_sample=4096,
            prompt=final_prompt,
            temperature=0.1
        )

        raw_response = completion.completion
        return parse_response_strip_assistant_message(raw_response=raw_response)



