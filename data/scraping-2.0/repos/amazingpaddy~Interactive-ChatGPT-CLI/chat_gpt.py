import asyncio
import time
from functools import wraps

import openai
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from openai.error import RateLimitError
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress

from config import CHAT_SETTINGS


def typing_animation_decorator(func):
    """
    Decorator for simulating a typing animation while waiting for the chat model's response.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        response_received = asyncio.Event()

        async def progress_animation():
            """
            An asynchronous function that updates the progress animation.
            """
            with Progress() as progress:
                typing_task = progress.add_task("[cyan]ChatGPT typing...")

                while not response_received.is_set():
                    progress.update(typing_task, advance=0.1)
                    await asyncio.sleep(0.2)

                progress.remove_task(typing_task)

        async def gather_response():
            """
            An asynchronous function that calls the decorated function and sets the response_received event.
            """
            resp = await func(*args, **kwargs)
            response_received.set()
            return resp

        animation_task = asyncio.create_task(progress_animation())
        response_task = asyncio.create_task(gather_response())

        response = await response_task
        await animation_task

        return response

    return wrapper


class ChatGPT:
    """
    ChatGPT is a class that provides an interface to interact with GPT-based models using OpenAI API.
    """

    def __init__(self, api_key, model_name: str):
        """
        Initializes a new ChatGPT instance.

        :param api_key: OpenAI API key.
        :param model_name: The name of the GPT-based model to use.
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.model_name = model_name
        self.console = Console()

    @typing_animation_decorator
    async def ask(self, prompt):
        """
        Asynchronously sends a prompt to the GPT model and returns the generated response.

        :param prompt: The user input to be processed by the GPT model.
        :return: The generated response from the GPT model.
        """
        completion = openai.ChatCompletion.create(model=self.model_name,
                                                  messages=[{"role": "user", "content": f"{prompt}"}],
                                                  max_tokens=CHAT_SETTINGS['max_tokens'],
                                                  temperature=CHAT_SETTINGS['temperature'])
        if completion:
            if 'error' in completion:
                return completion['error']['message']
            return completion.choices[0].message.content
        else:
            raise Exception("Exception occurred!!")

    async def ask_stream(self, prompt):
        """
        Asynchronously sends a prompt to the GPT model using streaming mode and prints the generated response.

        :param prompt: The user input to be processed by the GPT model.
        """
        try:
            response = openai.ChatCompletion.create(model=self.model_name,
                                                    messages=[{"role": "user", "content": f"{prompt}"}],
                                                    max_tokens=CHAT_SETTINGS['max_tokens'],
                                                    temperature=CHAT_SETTINGS['temperature'], stream=True)
            answer = ''
            for event in response:
                if answer:
                    self.console.print(answer, end='', markup=True, highlight=True, emoji=True)
                event_text = event['choices'][0]['delta']
                answer = event_text.get('content', '')
                time.sleep(0.01)
        except RateLimitError:
            self.console.print("RateLimit Error Occurred!! Try again.. ")

    @typing_animation_decorator
    async def ask_mem(self, prompt, conv_chain):
        """
        Asynchronously sends a prompt to the GPT model using memory management and returns the generated response.

        :param prompt: The user input to be processed by the GPT model.
        :param conv_chain: The conversation chain object used to manage memory in the conversation.
        :return: The generated response from the GPT model.
        """
        if prompt:
            return Markdown(conv_chain.run(input=prompt))

    async def ask_mem_stream(self, prompt, conv_chain: ConversationChain):
        """
        Asynchronously sends a prompt to the GPT model using memory management and prints the generated response.

        :param prompt: The user input to be processed by the GPT model.
        :param conv_chain: The conversation chain object used to manage memory in the conversation.
        """
        if prompt:
            req = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ])
            conv_chain.prompt = req
            await conv_chain.arun(input=prompt)
