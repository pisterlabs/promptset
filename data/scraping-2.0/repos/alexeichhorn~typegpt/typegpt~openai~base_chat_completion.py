from ..message_collection_builder import EncodedMessage
from .views import OpenAIChatModel
import tiktoken


class BaseChatCompletions:
    @staticmethod
    def max_tokens_of_model(model: OpenAIChatModel) -> int:
        match model:
            case "gpt-3.5-turbo" | "gpt-3.5-turbo-0301" | "gpt-3.5-turbo-0613":
                return 4096
            case "gpt-3.5-turbo-16k" | "gpt-3.5-turbo-16k-0613" | "gpt-3.5-turbo-1106":
                return 16384
            case "gpt-4" | "gpt-4-0314" | "gpt-4-0613":
                return 8192
            case "gpt-4-32k" | "gpt-4-32k-0314" | "gpt-4-32k-0613":
                return 32768
            case "gpt-4-1106-preview" | "gpt-4-vision-preview":
                return 128_000

    # copied from OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    @classmethod
    def num_tokens_from_messages(cls, messages: list[EncodedMessage], model: OpenAIChatModel | None = None) -> int:
        """Returns the number of tokens used by a list of messages."""
        if model is None:
            model = "gpt-3.5-turbo-0613"  # default model

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo":
            return cls.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif model == "gpt-3.5-turbo-16k":
            return cls.num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0613")
        elif model == "gpt-4":
            return cls.num_tokens_from_messages(messages, model="gpt-4-0613")
        elif model == "gpt-4-32k":
            return cls.num_tokens_from_messages(messages, model="gpt-4-32k-0613")
        elif model in ("gpt-3.5-turbo-0301"):
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model in (
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-32k-0314",
            "gpt-4-32k-0613",
            "gpt-4-1106-preview",
            "gpt-4-vision-preview",
        ):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
