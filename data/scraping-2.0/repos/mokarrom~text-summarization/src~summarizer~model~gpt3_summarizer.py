"""Interface with GPT-3 API."""
import os
import openai
import logging
import dotenv
from abc import ABC, abstractmethod
from openai import OpenAIError
from transformers import GPT2TokenizerFast
from summarizer.util import SRC_RESOURCES_DIR
from summarizer.aws_secrert_manager import get_openai_api_key_from_sm

TLDR_TAG = "\n\nTl;dr"
GPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")

logger = logging.getLogger(__name__)


class Summarizer(ABC):
    @abstractmethod
    def __init__(self, text_tokens: int, summary_tokens: int):
        self.text_tokens: int = text_tokens
        self.summary_tokens: int = summary_tokens

    def summarize(self, input_text) -> str:
        """Summarize the given text.
        Args:
            input_text: A large text.
        Returns:
            summary_text: The summary of the large text.
        """
        pass


class Gpt3Summarizer(Summarizer):
    def __init__(
            self,
            model_name: str,
            temperature: float,
            text_tokens: int,
            summary_tokens: int,
            top_p: float,
            frequency_penalty: float,
            presence_penalty: float
    ):
        super().__init__(text_tokens, summary_tokens)
        openai.api_key = Gpt3Summarizer.get_api_key()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        logger.info(f"OpenAI initialized with: model_name={model_name}, temperature={temperature}, "
                    f"text_tokens={text_tokens}, summary_tokens={summary_tokens}, top_p={top_p} "
                    f"frequency_penalty={frequency_penalty}, presence_penalty={presence_penalty}")

    def summarize(self, input_text) -> str:
        """Return summary of the given input_text."""
        logger.debug(f"############### Input Text ###############\n{input_text}\n.................................\n")
        try:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=input_text + TLDR_TAG,
                temperature=self.temperature,
                max_tokens=self.summary_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            summary_text: str = response["choices"][0]["text"]
        except OpenAIError as ex:
            logger.error(f"OpenAI Error: {ex.user_message}")
            raise RuntimeError("OpenAI Error", ex.user_message)
        logger.debug(f"############### Summary Text ###############\n{summary_text}\n..............................\n")
        return summary_text

    @classmethod
    def get_api_key(cls) -> str:
        api_key = dotenv.dotenv_values(os.path.join(SRC_RESOURCES_DIR, ".env"))["OPENAI_API_KEY"]
        if api_key == "None":
            api_key = get_openai_api_key_from_sm()
        else:
            logger.info("OpenAI API key retrieved from dotenv.")
        return api_key


def count_tokens(text: str):
    """Return the number of tokens in the given text."""
    return len(GPT2_TOKENIZER(text)["input_ids"])
