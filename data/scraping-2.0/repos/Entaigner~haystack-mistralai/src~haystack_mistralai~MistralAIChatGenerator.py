import dataclasses
import logging
import os
from typing import Optional, List, Callable, Dict, Any

# import openai
# from openai.openai_object import OpenAIObject

from mistralai.constants import ENDPOINT
from mistralai.client import MistralClient
from mistralai.models.chat_completion import (
    ChatMessage as MistralChatMessage,
    ChatCompletionStreamResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionResponse,
    ChatCompletionResponseChoice
)

from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.utils import serialize_callback_handler, deserialize_callback_handler
from haystack.dataclasses import StreamingChunk, ChatMessage

logger = logging.getLogger(__name__)


@component
class MistralAIChatGenerator:
    """
    Enables text generation using OpenAI's large language models (LLMs). It supports gpt-4 and gpt-3.5-turbo
    family of models accessed through the chat completions API endpoint.

    Users can pass any text generation parameters valid for the `openai.ChatCompletion.create` method
    directly to this component via the `**generation_kwargs` parameter in __init__ or the `**generation_kwargs`
    parameter in `run` method.

    For more details on the parameters supported by the OpenAI API, refer to the OpenAI
    [documentation](https://platform.openai.com/docs/api-reference/chat).

    ```python
    from haystack.components.generators.chat import GPTChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = GPTChatGenerator()
    response = client.run(messages)
    print(response)

    >>{'replies': [ChatMessage(content='Natural Language Processing (NLP) is a branch of artificial intelligence
    >>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
    >>meaningful and useful.', role=<ChatRole.ASSISTANT: 'assistant'>, name=None,
    >>metadata={'model': 'gpt-3.5-turbo-0613', 'index': 0, 'finish_reason': 'stop',
    >>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}

    ```

     Key Features and Compatibility:
         - **Primary Compatibility**: Designed to work seamlessly with the OpenAI API Chat Completion endpoint
            and gpt-4 and gpt-3.5-turbo family of models.
         - **Streaming Support**: Supports streaming responses from the OpenAI API Chat Completion endpoint.
         - **Customizability**: Supports all parameters supported by the OpenAI API Chat Completion endpoint.

     Input and Output Format:
         - **ChatMessage Format**: This component uses the ChatMessage format for structuring both input and output,
           ensuring coherent and contextually relevant responses in chat-based text generation scenarios. Details on the
           ChatMessage format can be found at: https://github.com/openai/openai-python/blob/main/chatml.md.
    """

    api_base: str | None = None
    model_name: str | None = None
    max_retries: int | None = None
    timeout: int | None = None
    generation_kwargs: Optional[Dict[str, Any]] = None

    client: MistralClient | None = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 5,
        timeout: int = 120,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Creates an instance of ChatGPTGenerator. Unless specified otherwise in the `model_name`, this is for OpenAI's
        GPT-3.5 model.

        :param api_key: The OpenAI API key. It can be explicitly provided or automatically read from the
            environment variable OPENAI_API_KEY (recommended).
        :param model_name: The name of the model to use.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param generation_kwargs: Other parameters to use for the model. These parameters are all sent directly to
            the OpenAI endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat) for
            more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
                it will generate two completions for each of the three prompts, ending up with 6 completions in total.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
                the model will be less likely to repeat the same token in the text.
            - `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
                Bigger values mean the model will be less likely to repeat the same token in the text.
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens, and the
                values are the bias to add to that token.
        """
        api_key = api_key or os.environ.get("MISTRAL_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "MistralAIGenerator expects an MistralAI API key. "
                "Set the MISTRAL_API_KEY environment variable (recommended) or pass it explicitly."
            )

        self.api_base = api_base or ENDPOINT
        self.model_name = model_name
        self.generation_kwargs = generation_kwargs or {}
        self.max_retries = max_retries
        self.timeout = timeout
        self.streaming_callback = streaming_callback

        self.client = MistralClient(
            api_key=api_key,
            endpoint=self.api_base,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        :return: The serialized component as a dictionary.
        """
        callback_name = serialize_callback_handler(
            self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model_name=self.model_name,
            streaming_callback=callback_name,
            api_base_url=self.api_base,
            generation_kwargs=self.generation_kwargs
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MistralAIChatGenerator":
        """
        Deserialize this component from a dictionary.
        :param data: The dictionary representation of this component.
        :return: The deserialized component instance.
        """
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callback_handler(
                serialized_callback_handler)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Additional keyword arguments for text generation. These parameters will
        potentially override the parameters passed in the __init__ method.
        For more details on the parameters supported by the OpenAI API, refer to the
        OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat/create).
        :return: A list containing the generated responses as ChatMessage instances.
        """

        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs,
                             **(generation_kwargs or {})}

        # adapt ChatMessage(s) to the format expected by the OpenAI API
        mistralai_messages = self._convert_to_openai_format(messages)

        completions: List[ChatMessage]
        if self.streaming_callback:
            chunks: List[StreamingChunk] = []

            for chunk in self.client.chat_stream(
                model=self.model_name,
                messages=mistralai_messages,
                **generation_kwargs
            ):
                if chunk.choices:
                    chunk_delta: StreamingChunk = self._build_chunk(
                        chunk, chunk.choices[0])
                    chunks.append(chunk_delta)
                    # invoke callback with the chunk_delta
                    self.streaming_callback(chunk_delta)

                completions = [self._connect_chunks(chunks)]
        else:
            completion: ChatCompletionResponse = self.client.chat(
                model=self.model_name,
                messages=mistralai_messages,
                **generation_kwargs
            )

            completions = [self._build_message(
                completion, choice) for choice in completion.choices]

        # before returning, do post-processing of the completions
        for completion in completions:
            self._check_finish_reason(completion)

        return {
            "replies": [message.content for message in completions],
            "metadata": [message.metadata for message in completions],
        }

    def _convert_to_openai_format(self, messages: List[ChatMessage]) -> List[MistralChatMessage]:
        """
        Converts List[ChatMessage]*1 to List[ChatMessage]*2, the format expected by the MistralAI API.
        *1 haystack ChatMessage (haystack\dataclasses\chat_message.py)
        *2 mistralai ChatMessage (mistralai\models\chat_completion.py) aliased to MistralChatMessage
        :param messages: The list of haystack ChatMessages.
        :return: The list of messages in the format expected by the MistralAI API.
        """
        formatted_messages: List[MistralChatMessage] = []
        for m in messages:
            formatted_messages.append(
                MistralChatMessage(role=m.role, content=m.content))
        return formatted_messages

    def _connect_chunks(self, chunks: List[StreamingChunk]) -> ChatMessage:
        """
        Connects the streaming chunks into a single ChatMessage.
        """
        complete_response = ChatMessage.from_assistant(
            "".join([chunk.content for chunk in chunks]))
        complete_response.metadata.update({
            "model": chunks[-1].model,
            "index": 0,
            "finish_reason": chunks[-1].choices[0].finish_reason,
            "usage": {},  # we don't have usage data for streaming responses
        })
        return complete_response

    def _build_message(self,
                       completion: ChatCompletionResponse,
                       choice: ChatCompletionResponseChoice
                       ) -> ChatMessage:
        """
        Converts the response from the MistralAI API to a ChatMessage.
        :param completion: The completion returned by the MistralAI API.
        :param choice: The choice returned by the MistralAI API.
        :return: The ChatMessage.
        """
        message: ChatMessage = choice.message
        content = message.content
        chat_message = ChatMessage.from_assistant(content)
        chat_message.metadata.update(
            {
                "model": completion.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "usage": vars(completion.usage),
            }
        )
        return chat_message

    def _build_chunk(self,
                     chunk: ChatCompletionStreamResponse,
                     choice: ChatCompletionResponseStreamChoice
                     ) -> StreamingChunk:
        """
        Converts the response from the MistralAI API to a StreamingChunk.
        :param chunk: The chunk returned by the MistralAI API.
        :param choice: The choice returned by the MistralAI API.
        :return: The StreamingChunk.
        """
        has_content = bool(hasattr(choice.delta, "content")
                           and choice.delta.content)

        chunk_message = StreamingChunk(
            content=choice.delta.content if has_content else ""
        )
        chunk_message.metadata.update({
            "model": chunk.model,
            "index": choice.index,
            "finish_reason": choice.finish_reason
        })
        return chunk_message

    def _check_finish_reason(self, message: ChatMessage) -> None:
        """
        Check the `finish_reason` returned with the MistralAI completions.
        If the `finish_reason` is `length`, log a warning to the user.
        :param message: The message returned by the LLM.
        """
        if message.metadata["finish_reason"] == "length":
            logger.warning(
                "The completion for index %s has been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions.",
                message.metadata["index"],
            )
        if message.metadata["finish_reason"] == "model_length":
            logger.warning(
                "The completion for index %s has been truncated due to reaching the models context length.", message.metadata[
                    "index"]
            )
