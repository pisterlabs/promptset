import asyncio
import re
import queue
from abc import ABC, abstractmethod
from pathlib import Path

import openai

from robojudge.utils.logger import logging
from robojudge.utils.async_tools import make_async, gather_with_concurrency
from robojudge.utils.settings import settings
from robojudge.utils.gpt_tokenizer import tokenizer

openai.api_key = settings.OPENAI_API_KEY
openai.api_type = "azure"
openai.api_base = settings.OPENAI_API_BASE
openai.api_version = "2023-03-15-preview"

logger = logging.getLogger(__name__)

# TODO: limit output token count in prompt and with param
# TODO: how to handle if there are too many summaries to summarize?

# Measured in tokens
MAX_CONTEXT_SIZE = 4000
CZ_SPLIT_REGEX = re.compile(r"(?<![0-9])\.\s(?![a-z0-9ěščřžýáíéůú])")

DEFAULT_FILE_NAME = 'default.txt'
CHUNK_PATH = Path('datasets/llm-summary-testing/chunks/')
CHUNK_SUMMARY_PATH = Path('datasets/llm-summary-testing/piecewise-summaries/')

RESULT_SUMMARY_SYSTEM_MESSAGE = """
    You are a legal assistant who creates a summary of a court ruling.
    You will receive several paragraphs summarizing parts of the ruling.
    Put these paragraphs together into a single coherent summary. The summary should be between 5 and 15 sentences long.
    Answer ONLY in Czech.
"""

class BaseSummarizer(ABC):
    llm_type: str

    def __init__(self, text: str, file_name: str, context_size=MAX_CONTEXT_SIZE) -> None:
        self.text = text
        self.file_name = file_name if file_name else DEFAULT_FILE_NAME
        self.safe_context_size = int(context_size - (context_size * 0.15))

    async def summarize_text(self, cache_chunks=False, cache_chunk_summaries=False, create_overall_summary = True) -> str:
        chunk_path = CHUNK_PATH / self.llm_type / self.file_name
        if cache_chunks and chunk_path.exists():
            self.chunks = chunk_path.read_text().split('\n')
        else:
            self.chunks = Chunker.chunk_text(self.text, self.safe_context_size)
            if cache_chunks:
                self.save_chunks()

        chunk_summary_path = CHUNK_SUMMARY_PATH / self.llm_type / self.file_name
        if cache_chunk_summaries and chunk_summary_path.exists():
            self.chunk_summaries = chunk_summary_path.read_text().split('\n')
        else:
            self.chunk_summaries = await self.create_chunk_summaries()
            if cache_chunk_summaries:
                self.save_chunk_summaries()

        if len(self.chunk_summaries) == 1:
            logger.info(
                'The text fit into a single chunk, returning its summary directly.')
            return self.chunk_summaries[0]

        if create_overall_summary:
            return await self.summarize_summaries()
        
        return ''

    async def create_chunk_summaries(self) -> list[str]:
        coroutines = map(self.summarize_text_chunk, self.chunks)
        return await gather_with_concurrency(settings.SUMMARIZE_MAX_PARALLEL_REQUESTS, *coroutines)

        # TODO: previous summaries if they are useful for context
        # coroutines = []
        # for chunk_index, chunk in enumerate(self.chunks):
        #     previous_summary = ''
        #     if chunk_index >= 1:
        #         previous_summary = chunk_summaries[chunk_index-1]

        #     coroutines.append(self.summarize_text_chunk(chunk))

        # return await asyncio.gather(*coroutines)

    @classmethod
    def ensure_path(cls, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    def save_chunks(self):
        chunk_path = CHUNK_PATH / self.llm_type / self.file_name
        BaseSummarizer.ensure_path(chunk_path)
        with open(chunk_path, 'w') as wf:
            wf.writelines('\n'.join(self.chunks))

    def save_chunk_summaries(self):
        chunk_summary_path = CHUNK_SUMMARY_PATH / self.llm_type / self.file_name
        BaseSummarizer.ensure_path(chunk_summary_path)
        with open(chunk_summary_path, 'w') as wf:
            wf.writelines('\n'.join(self.chunk_summaries))

    @abstractmethod
    @make_async
    def summarize_text_chunk(self, text_chunk: str):
        ...

    @make_async
    def summarize_summaries(self):
        messages = [
            {"role": "system", "content": RESULT_SUMMARY_SYSTEM_MESSAGE},
            {"role": "user", "content": "\n".join(self.chunk_summaries)},
        ]

        try:
            chat_completion = openai.ChatCompletion.create(
                engine=settings.GPT_MODEL_NAME, messages=messages, temperature=0
            )

            return chat_completion.choices[0].message.content
            # return 'Summary:' + "\n".join(self.chunk_summaries)
        except Exception:
            logging.exception("Exception while calling OpenAI API:")

        return ""


class Chunker:
    SYSTEM_MESSAGE = """
You are an AI assistant that decides where to split text in half so that the halves are as semantically meaningful as possible.
Return ONLY the index of the first sentence in the second half of text and nothing else.
"""

    USER_MESSAGE = """
Text to split: {text}
"""

    @classmethod
    def split_text_into_sentences(cls, text: str) -> list[str]:
        sentences = []
        sentence_split = re.split(CZ_SPLIT_REGEX, text)
        for sentence in sentence_split:
            sentence = sentence.strip("").replace("\n", " ") + "."
            sentences.append(sentence)
        return sentences

    @classmethod
    def chunk_text(cls, text: str, safe_context_size=MAX_CONTEXT_SIZE):
        chunks: list[str] = []
        infinite_check = 0
        text_in_max_tokens = int(
            (len(tokenizer.encode(text)) / safe_context_size) * 2)

        text_sentences = cls.split_text_into_sentences(text)

        Q = queue.Queue()
        for sentence in text_sentences:
            Q.put(sentence)

        current_sentences_token_count = 0
        current_chunk_sentences = []
        while not Q.empty() and infinite_check < text_in_max_tokens:
            current_sentence = Q.get()
            current_sentence_token_count = Chunker.get_token_count(
                current_sentence)

            if current_sentences_token_count + current_sentence_token_count < safe_context_size:
                current_sentences_token_count += current_sentence_token_count
                current_chunk_sentences.append(current_sentence)
            else:
                batch = " ".join(current_chunk_sentences)
                sentence_index = int(cls.decide_split(batch))

                logger.debug(
                    f'Sentence index for next split: {sentence_index}')

                chunks.append(
                    ' '.join(current_chunk_sentences[:sentence_index]))

                # Reset
                del current_chunk_sentences[:sentence_index]
                current_sentences_token_count = Chunker.get_token_count(
                    ' '.join(current_chunk_sentences))
                current_sentences_token_count += current_sentence_token_count
                current_chunk_sentences.append(current_sentence)

                infinite_check += 1

        logger.debug(f'There were {infinite_check} splits decided by LLM.')

        if len(current_chunk_sentences):
            chunks.append(' '.join(current_chunk_sentences))

        return chunks

    # @classmethod
    # def decide_split(cls, text: str):
    #     return len(tokenizer.encode(text)) -1

    @classmethod
    def decide_split(cls, text: str):
        messages = [
            {"role": "system", "content": cls.SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": cls.USER_MESSAGE.format(
                    text=text
                ),
            },
        ]

        try:
            chat_completion = openai.ChatCompletion.create(
                engine="chatgpt", messages=messages, temperature=0
            )

            response = chat_completion.choices[0].message.content

            pattern = re.compile(r'(\d+)')
            return int(re.search(pattern, response).group(0))
        except Exception:
            logging.exception("Exception while calling OpenAI API:")

    @staticmethod
    def get_token_count(text: str) -> int:
        num_tokens = len(tokenizer.encode(text))
        return num_tokens
