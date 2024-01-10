"""Async Simple OpenAI API wrapper

The is the async version of the Simple OpenAI API wrapper which uses the [`aiohttp`](https://docs.aiohttp.org/en/stable/index.html) library.

It is intended for use with asyncio applications.  If you are not using asyncio, you should use the [Simple OpenAI API wrapper](simple_openai.md) instead.
"""

from pathlib import Path
from typing import Callable
import aiohttp

from . import constants
from .models import open_ai_models
from .responses import SimpleOpenaiResponse
from . import chat_manager
from . import function_manager

class AsyncSimpleOpenai:
    """Async Simple OpenAI API wrapper

    This class implements the Async Simple OpenAI API wrapper.

    To use this class, you need to have an OpenAI API key. You can get one from [Openai](https://platform.openai.com).

    An optional storage path can be provided.  If a storage path is provided, the chat messages will be stored in the directory specified by the storage path.  If no storage path is provided, the chat messages will not be stored.

    Args:
        api_key (str): Your OpenAI API key
        system_message (str): The system message to add to the start of the chat
        storage_path (Path, optional): The path to the storage directory. Defaults to None.
        timezone (str, optional): The timezone to use for the chat messages. Defaults to 'UTC'.

    !!!Example
        ```python
        from simple_openai import AsyncSimpleOpenai
        import asyncio

        async def main():
            # Get the storage path
            storage_path = Path("/path/to/storage")

            # Create a system message
            system_message = "You are a helpful chatbot. You are very friendly and helpful. You are a good friend to have."

            # Create the client
            client = AsyncSimpleOpenai(api_key, system_message, storage_path)

            # Create tasks for the chat response and the image response
            tasks = [
                client.get_chat_response("Hello, how are you?", name="Bob", chat_id="Group 1"),
                client.get_image_url("A cat"),
            ]

            # Wait for the tasks to complete
            for task in asyncio.as_completed(tasks):
                # Get the result
                result = await task

                # Print the result
                if result.success:
                    # Print the message
                    print(f'Success: {result.message}')
                else:
                    # Print the error
                    print(f'Error: {result.message}')

        if __name__ == "__main__":
            # Run the main function
            asyncio.run(main())
        ```
    """
    def __init__(self, api_key: str, system_message: str, storage_path: Path | None = None, timezone: str = 'UTC') -> None:
        self._headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        # Create the chat manager
        self._chat = chat_manager.ChatManager(system_message, storage_path=storage_path, timezone=timezone)

        # Create the function manager
        self._function_manager = function_manager.FunctionManager()

    def update_system_message(self, system_message: str) -> None:
        """Update the system message

        Args:
            system_message (str): The new system message
        """
        self._chat.update_system_message(system_message)

    def add_function(self, function_definition: open_ai_models.OpenAIFunction, function: Callable) -> None:
        """Add a function to the function manager

        Args:
            function_definition (open_ai_models.OpenAIFunction): The function definition
            function (Callable): The function to call
        """
        self._function_manager.add_function(function_definition, function)

    async def get_chat_response(self, prompt: str, name: str, chat_id: str = constants.DEFAULT_CHAT_ID, add_date_time: bool = False) -> SimpleOpenaiResponse:
        """Get a chat response from OpenAI

        An optional chat ID can be provided.  If a chat ID is provided, the chat will be continued from the chat with the specified ID.  If no chat ID is provided, all messages will be mixed into a single list.

        Args:
            prompt (str): The prompt to use for the chat response
            name (str): The name of the user
            chat_id (str, optional): The ID of the chat to continue. Defaults to DEFAULT_CHAT_ID.
            add_date_time (bool, optional): Whether to add the date and time to the message. Defaults to False.

        Returns:
            SimpleOpenaiResponse: The chat response, the value of `success` should be checked before using the value of `message`

        """
        # Add the message to the chat
        messages = self._chat.add_message(open_ai_models.ChatMessage(role='user', content=prompt, name=name), chat_id=chat_id, add_date_time=add_date_time).messages        

        # Create the request body
        request_body = open_ai_models.ChatRequest(messages=messages, functions=self._function_manager.get_json_function_list(), function_call='auto')

        # Delete the functions from the request body if there are no functions
        if request_body.functions is None:
            del request_body.functions

        # Open a session
        async with aiohttp.ClientSession(headers=self._headers, base_url=constants.BASE_URL) as session:
            # Send the request
            async with session.post(constants.CHAT_URL, json=request_body.model_dump()) as response1:
                # Check the status code
                if response1.status == 200:
                    # Get the response content
                    response_text = await response1.text()

                    # Parse the response body
                    response_body = open_ai_models.ChatResponse.model_validate_json(response_text)

                    # Check if a function was called
                    if response_body.choices[0].finish_reason == constants.OPEN_AI_FUNCTION_CALL and response_body.choices[0].message.function_call is not None:
                        # Call the function
                        new_prompt = await self._function_manager.async_call_function(response_body.choices[0].message.function_call.name)

                        # Add the response to the chat
                        self._chat.add_message(open_ai_models.ChatMessage(role='assistant', content=response_body.choices[0].message.function_call.model_dump_json(), name='Botto'))

                        # Add the message to the chat
                        messages = self._chat.add_message(open_ai_models.ChatMessage(role='function', content=new_prompt, name='Botto'), chat_id=chat_id, add_date_time=add_date_time).messages

                        # Create the request body
                        request_body = open_ai_models.ChatRequest(messages=messages, functions=self._function_manager.get_json_function_list(), function_call='none')

                        # Send the request
                        async with session.post(constants.CHAT_URL, json=request_body.model_dump()) as response2:
                            # Check the status code
                            if response2.status == 200:
                                # Get the response content
                                response_text = await response2.text()

                                # Parse the response body
                                response_body = open_ai_models.ChatResponse.model_validate_json(response_text)

                                # Create the response
                                if response_body.choices[0].message.content is not None:
                                    open_ai_response = SimpleOpenaiResponse(True, response_body.choices[0].message.content)
                                else:
                                    open_ai_response = SimpleOpenaiResponse(True, 'No response')

                                # Add the response to the chat
                                self._chat.add_message(open_ai_models.ChatMessage(role='assistant', content=open_ai_response.message, name='Botto'))
                            else:
                                # Parse the error response body
                                response_body = open_ai_models.ErrorResponse.model_validate_json(await response2.text())

                                # Create the response
                                open_ai_response = SimpleOpenaiResponse(False, response_body.error.message)
                    else:
                        # Create the response
                        if response_body.choices[0].message.content is not None:
                            open_ai_response = SimpleOpenaiResponse(True, response_body.choices[0].message.content)
                        else:
                            open_ai_response = SimpleOpenaiResponse(True, 'No response')

                        # Add the response to the chat
                        self._chat.add_message(open_ai_models.ChatMessage(role='assistant', content=open_ai_response.message, name='Botto'))
                else:
                    # Parse the error response body
                    response_body = open_ai_models.ErrorResponse.model_validate_json(await response1.text())

                    # Create the response
                    open_ai_response = SimpleOpenaiResponse(False, response_body.error.message)

                # Return the response
                return open_ai_response

    async def get_image_url(self, prompt: str) -> SimpleOpenaiResponse:
        """Get an image response from OpenAI

        Args:
            prompt (str): The prompt to use

        Returns:
            SimpleOpenaiResponse: The image response, the value of `success` should be checked before using the value of `message`
        """
            
        # Create the request body
        request_body = open_ai_models.ImageRequest(prompt=prompt)

        # Open a session
        async with aiohttp.ClientSession(headers=self._headers, base_url=constants.BASE_URL) as session:
            # Send the request
            async with session.post(constants.IMAGE_URL, json=request_body.model_dump()) as response:
                # Check the status code
                if response.status == 200:
                    # Parse the response body
                    response_body = open_ai_models.ImageResponse.model_validate_json(await response.text())

                    # Create the response
                    response = SimpleOpenaiResponse(True, response_body.data[0].url)
                else:
                    # Parse the error response body
                    response_body = open_ai_models.ErrorResponse.model_validate_json(await response.text())

                    # Create the response
                    response = SimpleOpenaiResponse(False, response_body.error.message)

                # Return the response
                return response
