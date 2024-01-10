import asyncio
import re
import queue

import openai

from robojudge.components.summarizer.base_summarizer import BaseSummarizer, openai
from robojudge.utils.logger import logging
from robojudge.utils.async_tools import make_async
from robojudge.utils.settings import settings

logger = logging.getLogger(__name__)

CHUNK_SUMMARY_SYSTEM_MESSAGE = """
    You are a legal assistant who summarizes the provided court ruling.
    You will receive a part of the court ruling that you should summarize
    Summarize what this part of the court ruling is about based on the text and what preceded it.
    Create your summary ONLY in Czech. Return only the summary.
"""

CHUNK_SUMMARY_USER_MESSAGE = """
Text to summarize: {text}
"""


# TODO: limit output token count in prompt and with param
# TODO: how to handle if there are too many summaries to summarize?


class GPTSummarizer(BaseSummarizer):
    llm_type = 'gpt'
    MAX_CONTEXT_SIZE = 4000

    def __init__(self, text: str, file_name: str = '') -> None:
        super().__init__(text, file_name, context_size=self.MAX_CONTEXT_SIZE)

    @make_async
    def summarize_text_chunk(self, text_chunk: str):
        messages = [
            {"role": "system", "content": CHUNK_SUMMARY_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": CHUNK_SUMMARY_USER_MESSAGE.format(
                    text=text_chunk
                ),
            },
        ]

        try:
            chat_completion = openai.ChatCompletion.create(
                engine=settings.GPT_MODEL_NAME, messages=messages, temperature=0
            )

            return chat_completion.choices[0].message.content
            # return 'Summary:' + text_chunk
        except Exception:
            logging.exception("Exception while calling OpenAI API:")

        return ""
