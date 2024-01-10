import logging
from threading import Thread
from typing import Generator, Generic, List, Tuple, TypeVar

import openai
import openai.util
import tiktoken
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextGenerationPipeline,
    TextIteratorStreamer
)

T = TypeVar("T")


class TextGenInference(Generic[T]):
    def __init__(self, **kwargs) -> None:
        pass

    def generate_from_prompt(
        self,
        prompt: str,
        context: T,
        **kwargs
    ) -> Generator[str, T, None] | Tuple[str, T]:
        pass

    def encode(self, string: str) -> List[int]:
        pass

    def decode(self, tokens: List[int]) -> str:
        pass

    def trim_context(self, context: T, token_limit: int = 1_000) -> T:
        pass


class TextGenProcessor:
    def __init__(self, model: str | TextGenInference, **kwargs) -> None:
        self.logger = kwargs.pop("logger", None) or logging.getLogger("text_gen")

        self.max_context_tokens: int = kwargs.pop("max_context_tokens", 100)
        self.context = kwargs.pop("context", None)

        # Initialize inference
        self.inference: TextGenInference = None

        if isinstance(model, str):
            if model.startswith("text-") and "code" not in model:
                self.inference = OpenAI(model, **kwargs)
            elif model.startswith("gpt-3.5") or model.startswith("gpt-4"):
                self.inference = OpenAIChat(model, **kwargs)
            else:
                self.inference = TransformersInference(model, **kwargs)
        else:
            self.inference = model

        self.logger.info(
            "Text Gen Processor initialized with model `%s`.",
            self.inference.__class__.__name__
        )

    def generate_from_prompt(self, prompt: str, **kwargs) -> Generator[str, None, None] | str:
        # Add the max number of tokens from context to prompt
        if kwargs.pop("with_context", True):
            self.trim_context()
            kwargs["context"] = self.context

        # Handle adding new text to context while streaming
        if kwargs.get("stream", False):
            # This generator will just yield whatever generator_from_prompt,
            # but place the return value in self.context
            # TODO: I dont think this is thread safe
            def generator():
                self.context = yield from self.inference.generate_from_prompt(prompt, **kwargs)
            return generator()

        # Add new text to context
        generated_text, self.context = self.inference.generate_from_prompt(prompt, **kwargs)
        return generated_text

    def trim_context(self):
        self.context = self.inference.trim_context(
            self.context,
            token_limit=self.max_context_tokens
        )

    def clear_context(self):
        self.context = None


class OpenAI(TextGenInference[str]):
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = model

        if "openai_api_key" not in kwargs:
            self.api_key = openai.util.default_api_key()
        else:
            self.api_key = kwargs.pop("openai_api_key")

        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Most likely a fine tune, default to r50k_base
            self.tokenizer = tiktoken.get_encoding("r50k_base")

    def generate_from_prompt(
        self,
        prompt: str,
        context: str,
        **kwargs
    ) -> Generator[str, str, None] | Tuple[str, str]:
        # Add model, api key, and prompt into request
        if "model" not in kwargs:
            kwargs["model"] = self.model

        if "api_key" not in kwargs:
            kwargs["api_key"] = self.api_key

        prompt = context + prompt
        kwargs["prompt"] = prompt

        # Return a generator that just extracts the content of each event if streaming
        if kwargs.get("stream", False):
            def generator():
                # A list to store newly generated text
                chunks = [prompt]
                for event in openai.Completion.create(**kwargs):
                    chunks.append(event["choices"][0]["text"])
                    yield event["choices"][0]["text"]

                # Return full new text
                return "".join(chunks)
            return generator()

        res = openai.Completion.create(**kwargs)["choices"][0]["text"]
        return (res, context + res)

    def encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def trim_context(self, context: str, token_limit: int = 1_000) -> str:
        # Ensure context is a string
        if context is None:
            context = ""
        elif not isinstance(context, str):
            raise ValueError("OpenAI inference context must be a string.")

        # Trim tokens by converting to tokens and just taking however many
        # tokens from the end of the tokens and decode that into text
        tokens = self.encode(context)
        return self.decode(tokens[-token_limit:])


class OpenAIChat(TextGenInference[list]):
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = model

        if "openai_api_key" not in kwargs:
            self.api_key = openai.util.default_api_key()
        else:
            self.api_key = kwargs.pop("openai_api_key")

        self.tokenizer = tiktoken.encoding_for_model(model)

    def generate_from_prompt(
        self,
        prompt: str,
        context: list,
        **kwargs
    ) -> Generator[str, list, None] | Tuple[str, list]:
        # Add model, api key, and messages into request
        if "model" not in kwargs:
            kwargs["model"] = self.model

        if "api_key" not in kwargs:
            kwargs["api_key"] = self.api_key

        # Don't want to modify in place, copy instead
        messages = context.copy()
        messages.append({"role": "user", "content": prompt})
        kwargs["messages"] = messages

        # Return a generator that just extracts the content of each event if streaming
        if kwargs.get("stream", False):
            def generator():
                chunks = []
                role = None

                # The first event will return the name of the role,
                # the subsequent events will be the actual text
                for event in openai.ChatCompletion.create(**kwargs):
                    delta = event["choices"][0]["delta"]
                    if "role" in delta:
                        role = delta["role"]
                    elif "content" in delta:
                        chunks.append(delta["content"])
                        yield delta["content"]

                messages.append({"role": role, "content": "".join(chunks)})
                # Return full new text
                return messages
            return generator()

        res = openai.ChatCompletion.create(**kwargs)["choices"][0]["message"]
        messages.append(res)
        return (res["content"], messages)

    def encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def trim_context(self, context: list, token_limit: int = 1_000) -> list:
        # Ensure context is a list
        if context is None:
            context = []
        elif not isinstance(context, list):
            raise ValueError("OpenAI chat inference context must be a list.")

        # Trim tokens by counting tokens from newest to oldest, then
        # removing any message that goes over. So it won't trim exactly to the limit,
        # but close enough
        # Based off https://platform.openai.com/docs/guides/gpt/managing-tokens
        # TODO: Actually respect the limit rather than get "close enough" (ur dumb past me)
        token_count = 0
        res = []
        for message in reversed(context):
            token_count += 4
            for key, value in message.items():
                token_count += len(self.encode(value))
                if key == "name":
                    token_count -= 1
            token_count += 2

            if token_count <= token_limit:
                res.insert(0, message)
            else:
                break

        return res


class TransformersInference(TextGenInference[str]):
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        self.pipeline: TextGenerationPipeline = pipeline(
            task="text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=kwargs.get("device")
        )

        self.human_string = kwargs.pop("human_string", "")
        self.robot_string = kwargs.pop("robot_string", "")

    def generate_from_prompt(
        self,
        prompt: str,
        context: T,
        **kwargs
    ) -> Generator[str, T, None] | Tuple[str, T]:
        prompt = self.human_string + prompt + self.robot_string
        print(context)

        # Return the streamer and start the pipeline in a diff thread
        if kwargs.pop("stream", False):
            # While I don't like making a new thread, it seems this is necessary
            Thread(name="generation_thread", target=self.pipeline, args=(prompt,), kwargs={
                "prefix": context,
                "max_length": 1_000,
                "streamer": self.streamer,
                **kwargs
            }).start()
            def generator():
                # A list to store newly generated text
                chunks = [context, prompt]
    
                for word in self.streamer:
                    # TODO: This is a monkeypatch for llama2's eos(?) token. Fix this later
                    word = word.removesuffix("</s>")
                    chunks.append(word)
                    yield word

                # Return full new text
                return "".join(chunks)
            return generator()
        else:
            res = self.pipeline(
                prompt,
                return_full_text=False,
                prefix=context,
                max_length=1_000,
                **kwargs
            )
        text = res[0]["generated_text"]
        return (text, context + text)

    def encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def trim_context(self, context: str, token_limit: int = 1_000) -> str:
        # Ensure context is a string
        if context is None:
            context = ""
        elif not isinstance(context, str):
            raise ValueError("TransformersInference context must be a string.")

        # Trim tokens by converting to tokens and just taking however many
        # tokens from the end of the tokens and decode that into text
        tokens = self.encode(context)
        return self.decode(tokens[-token_limit:])


if __name__ == "__main__":
    with open("secrets", "r", encoding="utf-8") as secrets_file:
        OPENAI_API_KEY = secrets_file.readline().removesuffix("\n")
    processor = TextGenProcessor("georgesung/llama2_7b_chat_uncensored", human_string="\n\n### HUMAN:\n", robot_string="\n\n### RESPONSE:\n", max_context_tokens=1000, openai_api_key=OPENAI_API_KEY, device=0)
    processor.context = """Enter RP mode. Pretend to be a college frat boy.

You shall reply to the user while staying in character.
"""
    print(processor.context)
    while True:
        print("\n\n### HUMAN:\n", end="")
        thing = input()
        print("\n\n### RESPONSE:\n", end="")
        for word in processor.generate_from_prompt(thing, stream=True, repetition_penalty=1.2, length_penalty=0.8):
            print(word, end="")
    # for token in processor.generate_from_prompt(input("what") + ". respond with a series of messages separated by \"<eom>\"", stream=True):
    #     print(token, end="")


# pylint: disable-all
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, pipeline, StoppingCriteria


# class TextGenProcessor:
#     def __init__(self, model: str, device: str = "cuda") -> None:
#         self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model, low_cpu_mem_usage=True)
#         print(self.model.eval())

#         self.tokenizer = AutoTokenizer.from_pretrained(model)

#         # self.pipeline = pipeline(
#         #     task="text-generation",
#         #     model=self.model,
#         #     framework="pt",
#         #     device=torch.device(device),
#         #     stopping_criteria=_StoppingCriteria()
#         # )

#     def generate(self, prompt: str) -> str:
#         return ""


# class _StoppingCriteria(StoppingCriteria):
#     def __init__(self) -> None:
#         super().__init__()

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         print(input_ids)
#         return True


# if __name__ == "__main__":
#     processor = TextGenProcessor("Neko-Institute-of-Science/pygmalion-7b")
#     while True:
#         print(processor.generate(input("im so confused")))
