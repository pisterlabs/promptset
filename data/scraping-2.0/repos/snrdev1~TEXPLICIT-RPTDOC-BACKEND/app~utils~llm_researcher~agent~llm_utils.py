from __future__ import annotations

import json
import logging
from typing import List, Optional

import openai
from colorama import Fore, Style
from langchain.adapters import openai as lc_openai

from ..config import Config
from .prompts import auto_agent_instructions

CFG = Config()


def create_chat_completion(
    messages: list,  # type: ignore
    model: Optional[str] = None,
    temperature: float = CFG.temperature,
    max_tokens: Optional[int] = None,
    stream: Optional[bool] = False,
) -> str:
    """Create a chat completion using the OpenAI API
    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.
        stream (bool, optional): Whether to stream the response. Defaults to False.
    Returns:
        str: The response from the chat completion
    """

    # validate input
    if model is None:
        raise ValueError("Model cannot be None")
    if max_tokens is not None and max_tokens > 8001:
        raise ValueError(f"Max tokens cannot be more than 8001, but got {max_tokens}")
    if stream is None:
        raise ValueError("Websocket cannot be None when stream is True")

    # create response
    for attempt in range(10):  # maximum of 10 attempts
        response = send_chat_completion_request(
            messages, model, temperature, max_tokens, stream
        )
        return response

    logging.error("Failed to get response from OpenAI API")
    raise RuntimeError("Failed to get response from OpenAI API")


def send_chat_completion_request(messages, model, temperature, max_tokens, stream):
    if not stream:
        result = lc_openai.ChatCompletion.create(
            model=model,  # Change model here to use different models
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=CFG.llm_provider,  # Change provider here to use a different API
        )
        return result["choices"][0]["message"]["content"]
    else:
        return stream_response(model, messages, temperature, max_tokens)


async def stream_response(model, messages, temperature, max_tokens):
    paragraph = ""
    response = ""
    print(f"streaming response...")

    for chunk in lc_openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=CFG.llm_provider,
        stream=True,
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            response += content
            paragraph += content
            if "\n" in paragraph:
                print({"type": "report", "output": paragraph})
                paragraph = ""
    print(f"streaming response complete")
    return response


def choose_agent(task: str) -> dict:
    """Determines what agent should be used
    Args:
        task (str): The research question the user asked
    Returns:
        agent - The agent that will be used
        agent_role_prompt (str): The prompt for the agent
    """
    try:
        response = create_chat_completion(
            model=CFG.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{auto_agent_instructions()}"},
                {"role": "user", "content": f"task: {task}"},
            ],
            temperature=0,
        )

        return json.loads(response)
    except Exception as e:
        print(f"{Fore.RED}Error in choose_agent: {e}{Style.RESET_ALL}")
        return {
            "agent": "Default Agent",
            "agent_role_prompt": "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text.",
        }


async def llm_process_subtopics(task: str, subtopics: list) -> list:
    """
    The function `llm_process_subtopics` takes a main task and a list of subtopics as input, processes
    the subtopics according to certain rules, and returns the processed subtopics.

    Args:
      task (str): The `task` parameter represents the main topic or task that the subtopics are related
    to.
      subtopics (list): The `subtopics` parameter is a list of strings that represents the subtopics
    related to a main task. Each string in the list represents a subtopic.

    Returns:
      The function llm_process_subtopics returns a list of processed subtopics.
    """
    try:
        print(f"💎 Number of subtopics to be processed : {len(subtopics)}")

        prompt = f"""
            Provided the main topic -> {task}, subtopics -> {subtopics}
            - remove all generic subtopics containing tasks like: 
            '''
                - introduction
                - appendices
                - conclusion
                - overview
                - background
                - abstract
                - summary 
                - references
                etc.
            '''
            - merge subtopics closely related or similar in meaning while retaining the latest 'websearch' and 'source' values
            - Do NOT add any new subtopics
            - Retain the main task as the first subtopic
            - Limit the number of subtopics to a maximum of 10 (can be lower)
            - Finally order the subtopics by their tasks, in a relevant and meaningful order which is presentable in a detailed report
        """

        response = openai.ChatCompletion.create(
            model=CFG.fast_llm_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        output = json.loads(response.choices[0].message.content)["subtopics"]

        print(f"💎 Final number of subtopics : {len(output)}")

        return output

    except Exception as e:
        print("Exception in parsing subtopics : ", e)
        return subtopics
