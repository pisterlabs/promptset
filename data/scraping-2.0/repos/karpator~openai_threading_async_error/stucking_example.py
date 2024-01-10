import asyncio
from openai import AsyncAzureOpenAI
import threading
from typing import (
    Callable,
    Any,
)


"""
This class is used to trigger the OpenAI API stuck.
"""
def singleton(class_: Any) -> Any:
    """
    This is the singleton pattern, which is a software design pattern that restricts the instantiation of a class to
    one single instance. This is useful when exactly one object is needed to coordinate actions across the system.
    :param class_: any class type
    :return: none
    """
    __instances = {}

    def get_instance(*args: Any, **kwargs: Any) -> Any:
        """
        This function checks if there are any instances registered in the dictionary if not it will return a new
        instances of a class.
        :rtype Any
        :param args: non-keyword arguments
        :param kwargs: keyword arguments
        :return:
        """
        if class_ not in __instances:
            __instances[class_] = class_(*args, **kwargs)
        return __instances[class_]
    return get_instance


class AsyncThreadingHelper:
    def __init__(self, async_function: Callable):
        self.__async_function = async_function

    def __process_callback(self, *args: Any) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            self.__async_function(*args)
        )
        loop.close()

    def run_async_task_on_thread(self, args: Any) -> None:
        thread = threading.Thread(
            target=self.__process_callback,
            args=args
        )
        thread.start()


@singleton
class OpenAIAdapter:
    def __init__(self):
        self.__openai_client = AsyncAzureOpenAI(
            azure_endpoint=" AZURE ENDPOINT ",
            api_key=" API KEY ",
            max_retries=1,
            api_version="2023-05-15",
            timeout=10
        )

    async def get_chat_response(self, text: str = "How are you?"):
        _ = await self.__openai_client.chat.completions.create(
            messages=[{'role': 'system', 'content': "Generate a response for the user's message."},{'role': 'user', 'content': text}],
            model="gpt-4",
            max_tokens=800,
            presence_penalty=1.05,
            temperature=0,
            top_p=0.52,
            stream=False,
            timeout=10
        )


async def some_process_on_thread_a(openai_adapter: OpenAIAdapter):
    print("A: Start some process on thread A")
    await openai_adapter.get_chat_response()
    print("A: Finish some process on thread A")


async def some_process_on_thread_b(openai_adapter: OpenAIAdapter):
    print("B: Start some process on different thread")
    await openai_adapter.get_chat_response()
    print("B: Finish some process on different thread")


async def main():
    # Create OpenAI object on thread A
    openai_adapter = OpenAIAdapter()

    for i in range(10):
        print(f"Start iteration {i}")

        # Let's call an OpenAI call on thread B
        if i == 5:
            async_threading_helper = AsyncThreadingHelper(
                async_function=some_process_on_thread_b
            )
            async_threading_helper.run_async_task_on_thread(
                args=(
                    openai_adapter,
                )
            )

        # Call OpenAI on thread A
        await some_process_on_thread_a(openai_adapter)


if __name__ == '__main__':
    asyncio.run(main())
