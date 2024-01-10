from typing import Optional, Any, Union, List
import os
from dotenv import load_dotenv

import openai
from aitemplates.oai.ApiManager import SingleApiManager

from aitemplates.oai.utils.wrappers import retry_openai_api
from aitemplates.oai.types.chat import (
    ChatSequence,
    ChatConversation,
    Message,
)
from aitemplates.oai.types.functions import FunctionDef, Functions

dotenv_path = os.path.join(
    os.getcwd(), ".env"
)  # get the path to .env file in current working directory
load_dotenv(dotenv_path)  # load environment variables from the .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = os.getenv("model")

if OPENAI_API_KEY is None:
    raise Exception("API key not found in environment variables")

openai.api_key = OPENAI_API_KEY


@retry_openai_api()
def create_chat_completion(
    messages: Union[ChatSequence, ChatConversation, Message, List[Message], str],
    model: str = model or "gpt-3.5-turbo-0613",
    temperature: Optional[float] = 0,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = 1,
    n: Optional[int] = 1,
    stop: Optional[str] = None,
    presence_penalty: Optional[float] = 0,
    frequency_penalty: Optional[float] = 0,
    functions: Union[Optional[Functions], Optional[list[FunctionDef]]] = None,
    function_call: Optional[object] = None,
    send_object: bool = False,
    auto_call_func: bool = False,
) -> Any:
    """Create a chat completion using the OpenAI API

    Args:
        messages (list[MessageDict]): The messages to send to the chat completion.
        model (str, optional): The model to use. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): The temperature to use. Defaults to 0.
        max_tokens (int, optional): The maximum tokens to use. Defaults to None.
        top_p (float, optional): The nucleus sampling probability. Defaults to 1.
        n (int, optional): The number of messages to generate. Defaults to 1.
        stop (str, optional): The sequence at which the generation will stop. Defaults to None.
        presence_penalty (float, optional): The presence penalty to use. Defaults to 0.
        frequency_penalty (float, optional): The frequency penalty to use. Defaults to 0.
        functions (Optional[Functions], optional): The functions to use. Defaults to None.
        send_object (bool, optional): Whether to return the response object. Defaults to False.

    Returns:
        Any: The response from the chat completion.
    """
    kwarg_messages = None
    
    if isinstance(messages, str):
        messages = Message("system", messages)
        kwarg_messages = [messages.raw()]
    elif isinstance(messages, ChatConversation): # we set it to the last sequence which the response is None
        kwarg_messages = messages.conversation_history[-1].prompt.raw()
    elif isinstance(messages, Message):
        kwarg_messages = ChatSequence([messages]).raw()
        messages = ChatSequence([messages])
    elif isinstance(messages, List):
        kwarg_messages = ChatSequence(messages).raw()
        messages = ChatSequence(messages)
        kwarg_messages = messages.raw()
    else:
        kwarg_messages = messages.raw()

    api_manager = SingleApiManager()
    kwargs = {
        "model": model,
        "messages": kwarg_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "n": n,
        "stop": stop,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }

    process_functions = {"function_defs": None, "function_call": None}
    function_pairs = None
    
    if functions or (hasattr(messages, 'function_pairs') and len(messages.function_pairs)):
        process_functions = Functions.ensure_unique_functions(messages, functions)
    
    if process_functions.get("function_pairs"):
        function_pairs = process_functions.get("function_pairs")

    if process_functions.get("function_defs"):
        kwargs["functions"] = process_functions.get("function_defs")

    if function_call:
        kwargs["function_call"] = function_call

    response = openai.ChatCompletion.create(**kwargs)

    function_result = None

    if response.choices[0].message.get("function_call") and auto_call_func and function_pairs:
        function_result = Functions.execute_function_call(
            response.choices[0].message.function_call, function_pairs
        )

    api_manager.update_cost(
        response.usage.prompt_tokens, response.usage.completion_tokens, response.model
    )
    

    if isinstance(messages, ChatConversation):
        if function_result:
            # we want to set the content to the function_result
            messages.conversation_history[-1].update_response(response, function_result)
        else:
            messages.conversation_history[-1].update_response(response)

    if send_object:
        if function_result:
            return response, function_result
        else:
            return response
    elif n and n > 1:
        return response.choices
    elif function_result:
        return function_result
    else:
        return response.choices[0].message["content"]
