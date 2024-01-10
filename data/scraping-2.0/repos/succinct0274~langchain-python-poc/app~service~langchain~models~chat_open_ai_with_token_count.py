from typing import Any, AsyncIterator, Iterator, List, Optional
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import BaseMessage
from langchain.schema.output import ChatGenerationChunk
import tiktoken
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ChatOpenAIWithTokenCount(ChatOpenAI):

    def _stream(self, messages: List[BaseMessage], stop: List[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        self.validate_token_count(messages, stop)
        return super()._stream(messages, stop, run_manager, **kwargs)
    
    def _astream(self, messages: List[BaseMessage], stop: List[str] | None = None, run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        self.validate_token_count(messages, stop)
        return super()._astream(messages, stop, run_manager, **kwargs)
    
    def validate_token_count(self, messages: List[BaseMessage], stop: List[str]):
        message_dicts, params = self._create_message_dicts(messages, stop)
        count = self.num_tokens_from_messages(message_dicts, self.model_name)
        logger.info(f'The number of token used in this request: {count}')

    """
    Referenced from the following link: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    def num_tokens_from_messages(self, messages, model):
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                try:
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
                except Exception as error:
                    print(error)
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens