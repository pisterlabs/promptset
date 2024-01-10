from typing import Optional, List, Callable
from abc import ABC, abstractmethod
from .abstraction import OpenAIChatCompletionWrapper

class Handler(ABC):
    """All handlers must inherit from this class"""
    @abstractmethod
    def handle(self, source:OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        """
        This method executes the handler.

        After executing this method, the source object should be modified, and
        the response should be updated.
        """
        pass

# Retry Handlers
class DoNothing(Handler):
    def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        source.execute()
        return source

class IncreaseTemperatureRetry(Handler):
    def __init__(self, by: float=0.1, max_val: float=2.0):
        self.by = by
        self.max_val = max_val

    def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        source.temperature = min(source.temperature + self.by, self.max_val)
        source.execute()
        return source

class DecreaseTemperatureRetry(Handler):
    def __init__(self, by: float=0.1, min_val: float=0.0):
        self.by = by
        self.min_val = min_val

    def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        source.temperature = max(source.temperature - self.by, self.min_val)
        source.execute()
        return source

class IncreaseTopPRetry(Handler):
    def __init__(self, by: float=0.1, max_val: float=1.0):
        self.by = by
        self.max_val = max_val

    def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        source.top_p = min(source.top_p + self.by, self.max_val)
        source.execute()
        return source

class DecreaseTopPRetry(Handler):
    def __init__(self, by: float=0.1, min_val: float=0.0):
        self.by = by
        self.min_val = min_val

    def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        source.top_p = max(source.top_p - self.by, self.min_val)
        source.execute()
        return source

class IncreasePresencePenaltyRetry(Handler):
    def __init__(self, by: float=0.1, max_val: float=2.0):
        self.by = by
        self.max_val = max_val

    def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        source.presence_penalty = min(source.presence_penalty + self.by, self.max_val)
        source.execute()
        return source

class DecreasePresencePenaltyRetry(Handler):
    def __init__(self, by: float=0.1, min_val: float=-2.0):
        self.by = by
        self.min_val = min_val

    def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        source.presence_penalty = max(source.presence_penalty - self.by, self.min_val)
        source.execute()
        return source

class IncreaseFrequencyPenaltyRetry(Handler):
    def __init__(self, by: float=0.1, max_val: float=2.0):
        self.by = by
        self.max_val = max_val

    def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        source.frequency_penalty = min(source.frequency_penalty + self.by, self.max_val)
        source.execute()
        return source

class DecreaseFrequencyPenaltyRetry(Handler):
    def __init__(self, by: float=0.1, min_val: float=-2.0):
        self.by = by
        self.min_val = min_val

    def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        source.frequency_penalty = max(source.frequency_penalty - self.by, self.min_val)
        source.execute()
        return source

class AdditionalPromptRetry(Handler):
    def __init__(self, prompt: str = "Answer my query politely. "):
        self.prompt = prompt

    def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        source.messages.append(
            {"role": "user",
             "content":self.prompt}
        )
        source.execute()
        return source

# Censor Handlers
class Replace(Handler):
    def __init__(self,
                 keyword: List[str] | str,
                 replace_with: str="*"):
        if isinstance(keyword, str):
            keyword = [keyword]
        self.keyword = keyword
        self.replace_with = replace_with

    def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        for kw in self.keyword:
            source.response.choices[0].message.content = \
                source.response.choices[0].message.content.replace(kw, self.replace_with)
        return source

class Remove(Handler):
    def __init__(self, keyword: List[str] | str):
        if isinstance(keyword, str):
            keyword = [keyword]
        self.keyword = keyword

    def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        for kw in self.keyword:
            source.response.choices[0].message.content = \
                source.response.choices[0].message.content.replace(kw, "")
        return source


def custom_handler(handler: Callable[[OpenAIChatCompletionWrapper], OpenAIChatCompletionWrapper]) -> Handler:
    """
    This function converts a function into a handler.
    :param handler: A function that takes in a OpenAIChatCompletionWrapper and returns a OpenAIChatCompletionWrapper
    :return: A handler that executes the function

    Example:

    @custom_handler
    def my_handler(source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
        source.execute()
        return source
    """
    class CustomHandler(Handler):
        def handle(self, source: OpenAIChatCompletionWrapper) -> OpenAIChatCompletionWrapper:
            return handler(source)
    return CustomHandler()