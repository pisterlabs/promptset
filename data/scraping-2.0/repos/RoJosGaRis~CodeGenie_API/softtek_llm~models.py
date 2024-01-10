from abc import ABC, abstractmethod
from time import perf_counter_ns
from typing import Any, Literal, List

import openai
from typing_extensions import override

from softtek_llm.memory import Memory
from softtek_llm.schemas import Message, OpenAIChatResponse, Response, Filter
from softtek_llm.utils import setup_azure


class LLMModel(ABC):
    """
    # LLM Model
    Creates an abstract base class for a language model. Used as a base class for implementing different types of language models. Provides intialization with options like max_tokens, temperature and name. Also defines a call method that must be implemented.

    ## Parameters
    - `name`: Name of the model

    ## Methods
    - `__call__`: Method to generate text from the model. Must be implemented by the child class.
    """

    def __init__(
        self,
        model_name: str,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """Initializes the LLMModel class.

        Args:
            model_name (str): Name of the model
            verbose (bool, optional): Whether to print debug messages. Defaults to False.
        """
        super().__init__()
        self.__model_name = model_name
        self.__verbose = verbose

    @property
    def model_name(self) -> str:
        """Tthe name of the model."""
        return self.__model_name

    @property
    def verbose(self) -> bool:
        """Whether to print debug messages."""
        return self.__verbose

    @abstractmethod
    def __call__(
        self, memory: Memory, description: str = "You are a bot", **kwargs: Any
    ) -> Response:
        """
        A method to be overridden that calls the model to generate text.

        Args:
            memory (Memory): An instance of the Memory class containing the conversation history.
            description (str, optional): Description of the model. Defaults to "You are a bot".

        Returns:
            Response: The generated response.

        Raises:
        - NotImplementedError: When this abstract method is called without being implemented in a subclass.
        """
        raise NotImplementedError("__call__ method must be overridden")
    
    # TODO now: Change name?
    @abstractmethod
    def parse_filters(self, prompt: str) -> List[Message]:
        """
        Generates a prompt message to check if a given prompt follows a set of filtering rules.

        Args:
            prompt (str): a string representing the prompt that will be checked against rules

        Raises:
         - NotImplementedError: When this abstract method is called without being implemented in a subclass.
        """
        raise NotImplementedError("parse_filters method must be overriden")


class OpenAI(LLMModel):
    """
    # OpenAI
    Creates an OpenAI language model. This class is a subclass of the LLMModel abstract base class.

    ## Properties
    - `model_name`: Language model name.
    - `max_tokens`: The maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length.
    - `temperature`: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    - `presence_penalty`: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    - `frequency_penalty`: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.

    ## Methods
    - `__call__`: Method to generate text from the model.
    """

    @override
    def __init__(
        self,
        api_key: str,
        model_name: str,
        api_type: Literal["azure"] | None = None,
        api_base: str | None = None,
        api_version: str = "2023-07-01-preview",
        max_tokens: int | None = None,
        temperature: float = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        verbose: bool = False,
    ):
        """Initializes the OpenAI LLM Model class.

        Args:
            api_key (str): OpenAI API key.
            model_name (str): Name of the model.
            api_type (Literal["azure"] | None, optional): Type of API to use. Defaults to None.
            api_base (str | None, optional): Base URL for Azure API. Defaults to None.
            api_version (str, optional): API version for Azure API. Defaults to "2023-07-01-preview".
            max_tokens (int | None, optional): The maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length. Defaults to None.
            temperature (float, optional): What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. Defaults to 1.
            presence_penalty (float, optional): Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Defaults to 0.
            frequency_penalty (float, optional): Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Defaults to 0.
            verbose (bool, optional): Whether to print debug messages. Defaults to False.

        Raises:
            ValueError: When api_type is not "azure" or None.
        """
        super().__init__(model_name, verbose=verbose)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        openai.api_key = api_key
        if api_type is not None:
            openai.api_type = api_type
            match api_type:
                case "azure":
                    setup_azure(api_base, api_version)
                case _:
                    raise ValueError(
                        f"api_type must be either 'azure' or None, not {api_type}"
                    )

    @property
    def max_tokens(self) -> int | None:
        """The maximum number of tokens to generate in the chat completion.

        The total length of input tokens and generated tokens is limited by the model's context length.
        """
        return self.__max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int | None):
        if not isinstance(value, int) and value is not None:
            raise TypeError(
                f"max_tokens must be an integer or None, not {type(value).__name__}"
            )
        self.__max_tokens = value

    @property
    def temperature(self) -> float:
        """
        What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        """
        return self.__temperature

    @temperature.setter
    def temperature(self, value: float):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError(f"temperature must be a float, not {type(value).__name__}")

        if value < 0 or value > 2:
            raise ValueError("temperature must be between 0 and 2")
        self.__temperature = value

    @property
    def presence_penalty(self) -> float:
        """
        Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
        """
        return self.__presence_penalty

    @presence_penalty.setter
    def presence_penalty(self, value: float):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError(
                f"presence_penalty must be a float, not {type(value).__name__}"
            )
        if value < -2 or value > 2:
            raise ValueError("presence_penalty must be between -2 and 2")
        self.__presence_penalty = value

    @property
    def frequency_penalty(self) -> float:
        """
        Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        """
        return self.__frequency_penalty

    @frequency_penalty.setter
    def frequency_penalty(self, value: float):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError(
                f"frequency_penalty must be a float, not {type(value).__name__}"
            )
        if value < -2 or value > 2:
            raise ValueError("frequency_penalty must be between -2 and 2")
        self.__frequency_penalty = value

    @override
    def __call__(self, memory: Memory, description: str = "You are a bot.") -> Response:
        """
        Process a conversation using the OpenAI model and return a Response object.

        This function sends a conversation stored in the 'memory' parameter to the specified OpenAI model
        (self.model_name), retrieves a response from the model, and records the conversation in 'memory'.
        It then constructs a Response object containing the model's reply.

        Args:
            memory (Memory): An instance of the Memory class containing the conversation history.
            description (str, optional): Description of the model. Defaults to "You are a bot.".

        Returns:
            Response: A Response object containing the model's reply, timestamp, latency, and model name.
        """

        start = perf_counter_ns()
        messages = [message.model_dump() for message in memory.get_messages()]
        messages.insert(0, Message(role="system", content=description).model_dump())

        if self.verbose:
            print(f"Memory: {messages}")

        answer = OpenAIChatResponse(
            **openai.ChatCompletion.create(
                deployment_id=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
            )
        )

        resp = Response(
            message=Message(
                role="assistant", content=answer.choices[0].message.content
            ),
            created=answer.created,
            latency=(perf_counter_ns() - start) // 1e6,
            from_cache=False,
            model=answer.model,
            usage=answer.usage,
        )

        memory.add_message(resp.message.role, resp.message.content)

        return resp

    @override
    def parse_filters(self, prompt: str, context: List[Message], filters: List[Filter]) -> List[Message]:
        """
        Generates a prompt message to check if a given prompt follows a set of filtering rules.

        Args:
            prompt (str): a string representing the prompt that will be checked against rules.
            context (List[Message]): A list containing the last 3 messages from the chat.
            filters (List[Filter]): List of filters used by the chatbot.

        Returns:
            (List[Message]): a list of messages to be used by the chatbot to check if the prompt respects the rules
        """
        context = "\n".join(
            [f"\t- {message.role}: {message.content}" for message in context]
        )

        rules = "\n".join(
            [f"\t- {filter.type}: {filter.case}" for filter in filters]
        )

        prompt = f'Considering this context:\n{context}\n\nPlease review the prompt below and answer with "yes" if it adheres to the rules or "no" if it violates any of the rules.\nRules:\n{rules}\n\nPrompt:\n{prompt}'

        if self.verbose:
            print(f"Revision prompt: {prompt}")

        messages = [
            Message(**{"role": "system", "content": "only respond with 'yes' or 'no'"}),
            Message(**{"role": "user", "content": prompt}),
        ]

        return messages