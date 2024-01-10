from abc import ABC, abstractmethod
from inspect import signature
from faker import Faker
import time
from loguru import logger

try:
    import openai
except ImportError:
    logger.warning(
        "OpenAI API not installed, using fakeOpenAI - this is for testing only, it returns nonsense"
    )
    from robai.utility import interactiveOpenAI as openai
from openai import AsyncOpenAI
import os
from typing import Any, Union, Optional, List, Generator
from robai.in_out import ChatMessage
from robai.memory import BaseMemory
from robai.utility import fakeOpenAI


fake = Faker()


class BaseAIModel(ABC):
    # any_properties_for_the_ai_model, like max_tokens, temperature, etc
    # Set them here, and then use self.some_property in the call method

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def call(
        self,
        memory: BaseMemory,
    ) -> BaseMemory:
        """
        ### PARSES MEMORY.INSTRUCTIONS_FOR_AI AND SENDS TO THE AI MODEL
        What you must implement:
        * Parse the memory.instructions_for_ai,
        * Send the instructions to any AI model you like,
        * Add the raw response you get back to memory.ai_raw_response,
        * Add a ChatMessage[role='assistant', content='What the AI said'] to memory.ai_response
        * Return the memory object
        """
        # EXAMPLE IMPLEMENTATION
        # instructions_for_ai is _always_ a list of ChatMessage objects
        memory.instructions_for_ai: List[ChatMessage]
        # In OpenAI for example, the list of ChatMessage objects is almost exactly what
        # is needed, but they can't be sent as objects, they must be sent as dicts.
        correctly_parsed_instructions = [
            message.dict() for message in memory.instructions_for_ai
        ]
        response = fakeOpenAI.ChatCompletion.create(
            model="gpt-3.5-turbo-16k", messages=correctly_parsed_instructions
        )
        """
        This is what the raw response looks like from OpenAI:
        response = { 
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0613",
            "choices": [{
                "index": 0,
                "message": {
                "role": "assistant",
                "content": "\n\nHello there, how may I assist you today?",
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }
        """
        # We must store the raw response: maybe devs can use it in pre/postcall chains
        memory.ai_raw_response = response
        # IMPORTANT: we _must_  parse and store the response as a ChatMessage object.
        memory.ai_response = ChatMessage(
            role="assistant", content=response.choices[0]["message"]["content"]
        )
        # We are done - return the memory object

        return memory

    def stream_call(self, memory: BaseMemory) -> Generator:
        """
        You don't have to implement this, but in openai, we can have a generator returned
        so we can stream the response as it comes in.
        """
        pass

    async def acall(self, memory: BaseMemory) -> BaseMemory:
        pass

    async def astream_call(self, memory: BaseMemory) -> Generator:
        pass

    def call_manager(
        self, memory: BaseMemory, stream=False, **kwargs
    ) -> Union[BaseMemory, Generator]:
        """
        When you call process() on the robot, ulimately, this is the method that is called.
        In the very least it needs call() to be implemented.
        You probably don't want to override this method. But maybe there's a reason I don't know
        """
        if stream:
            memory = self.stream_call(memory=memory)
            return memory
        else:
            memory = self.call(memory=memory)
            return memory

    async def acall_manager(
        self, memory: BaseMemory, stream=False, **kwargs
    ) -> Union[BaseMemory, Generator]:
        """
        When you call process() on the robot, ulimately, this is the method that is called.
        In the very least it needs call() to be implemented.
        You probably don't want to override this method. But maybe there's a reason I don't know
        """
        if stream:
            memory = await self.astream_call(memory=memory)
            return memory
        else:
            memory = await self.acall(memory=memory)
            return memory

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Check if abstract methods in subclasses accept the same arguments as the base class
        base_methods = [BaseAIModel.call]
        for base_method in base_methods:
            base_signature = signature(base_method)
            subclass_method = getattr(cls, base_method.__name__)
            subclass_signature = signature(subclass_method)

            if base_signature.parameters != subclass_signature.parameters:
                raise TypeError(
                    f"Method {base_method.__name__} in subclass {cls.__name__} has an incorrect argument type. Look at the BaseAIModel.call method "
                    f"Expected {base_signature}, but got {subclass_signature}."
                )


class FakeAICompletion(BaseAIModel):
    instructions_for_ai: List[ChatMessage] = List[ChatMessage]

    def call(self, memory: BaseMemory) -> BaseMemory:
        # Here fake sentences would be the string response from the AI.
        fake_response = fake.sentences(nb=5)
        fake_response_string = " ".join(fake_response)
        # In any AI model, these are the two things you need to add to memory
        # The string response is the most important, have a look at what happens with get_ai_response()
        memory.ai_response = ChatMessage(role="assistant", content=fake_response_string)
        memory.ai_raw_response = fake_response
        # And now we return the memory object!
        return memory

    def stream_call(self, memory: BaseMemory) -> Generator:
        fake_response = " ".join(fake.sentences(nb=5))
        for word in fake_response.split(" "):
            streamed_response = word
            time.sleep(0.05)
            yield streamed_response


class OpenAICompletion(BaseAIModel):
    instructions_for_ai: str = str
    model: str = "gpt-3.5-turbo-16k"
    suffix: str = ""
    max_tokens: int = 1000
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    # logit_bias: Optional[Dict[str, float]] = None,
    user: Optional[str] = None
    openai = openai

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        API_KEY = os.getenv("OPENAI_API_KEY")
        if API_KEY is None:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.openai.api_key = API_KEY

    def call(
        self,
        memory: BaseMemory,
    ) -> BaseMemory:
        instructions_for_ai = [message.dict() for message in memory.instructions_for_ai]
        response = self.openai.Completion.create(
            model=self.model,
            prompt=instructions_for_ai,
            suffix=self.suffix,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            stream=self.stream,
            logprobs=self.logprobs,
            echo=self.echo,
            stop=self.stop,
            # presence_penalty=self.presence_penalty,
            # frequency_penalty=self.frequency_penalty,
            # best_of=self.best_of,
            # logit_bias=self.logit_bias,
            # user=self.user,
        )
        memory.ai_raw_response = response
        memory.ai_response = ChatMessage(
            role="assistant", content=response.choices[0].text.strip()
        )
        return memory


client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class OpenAIChatCompletion(OpenAICompletion):
    instructions_for_ai: List[ChatMessage] = List[ChatMessage]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, memory: BaseMemory) -> BaseMemory:
        instructions_for_ai = [
            message.model_dump() for message in memory.instructions_for_ai if message
        ]
        response = self.openai.ChatCompletion.create(
            model=self.model,
            messages=instructions_for_ai,
            temperature=self.temperature,
            stop=self.stop,
            max_tokens=self.max_tokens,
        )
        memory.ai_raw_response = response
        memory.ai_response = ChatMessage(
            role="assistant", content=response.choices[0]["message"]["content"]
        )
        return memory

    async def acall(self, memory: BaseMemory) -> BaseMemory:
        instructions_for_ai = [
            message.model_dump() for message in memory.instructions_for_ai if message
        ]
        response = await client.chat.completions.create(
            model=self.model,
            messages=instructions_for_ai,
            temperature=self.temperature,
            stop=self.stop,
            max_tokens=self.max_tokens,
        )
        memory.ai_raw_response = response
        memory.ai_response = ChatMessage(
            role="assistant", content=response.choices[0].message.content
        )

    def stream_call(self, memory: BaseMemory) -> Generator:
        instructions_for_ai = [
            message.dict() for message in memory.instructions_for_ai if message
        ]
        response_generator = self.openai.ChatCompletion.create(
            model=self.model,
            messages=instructions_for_ai,
            temperature=self.temperature,
            stop=self.stop,
            max_tokens=1000,
            stream=True,  # Enable streaming
        )

        memory.ai_response_generator = response_generator
        return memory

    async def astream_call(self, memory: BaseMemory) -> Generator:
        instructions_for_ai = [
            message.dict() for message in memory.instructions_for_ai if message
        ]
        response_generator = await client.chat.completions.create(
            model=self.model,
            messages=instructions_for_ai,
            temperature=self.temperature,
            stop=self.stop,
            max_tokens=1000,
            stream=True,  # Enable streaming
        )

        memory.ai_response_generator = response_generator
        return memory
