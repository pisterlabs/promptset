import openai
import streamlit as st
from typing import List, Dict, Tuple, Optional, Union, Callable
from .utils import count_tokens

# openai_api_key = st.secrets["OPENAI_API_KEY"]

# # Set the OpenAI API key
# openai.api_key = openai_api_key


def send_message_to_openai(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 1000,
    stream: bool = False,
) -> Dict[str, Union[str, List[str]]]:
    """
    Sends a message to OpenAI's model.

    Args:
        model (str): The model to use (e.g., "gpt-3.5-turbo").
        messages (List[Dict[str, str]]): The list of messages to send to the model.
        temperature (float, optional): Sampling temperature. Defaults to 0.3.
        max_tokens (int, optional): Maximum tokens for the response. Defaults to 1000.
        stream (bool, optional): Whether to stream the response. Defaults to False.

    Returns:
        Dict[str, Union[str, List[str]]]: The response from the model.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        return response
    except Exception as e:
        print(e)
        return {"error": f"GPT3 error: {e}"}


def answer_question(
    model: str = "gpt-4",
    instruction: str = "",
    prompt: List[Dict[str, str]] = None,
    temperature: float = 0.3,
    max_tokens: int = 1000,
    debug: bool = False,
) -> Tuple[str, int]:
    """
    Obtains an answer to a question from OpenAI's model.

    Args:
        model (str, optional): The model to use. Defaults to "gpt-3.5-turbo".
        instruction (str, optional): Instruction for the model. Defaults to provided instruction.
        prompt (List[Dict[str, str]], optional): The prompt for the question. Defaults to None.
        temperature (float, optional): Sampling temperature. Defaults to 0.3.
        max_tokens (int, optional): Maximum tokens for the response. Defaults to 1000.

    Returns:
        Tuple[str, int]: The answer text and the number of tokens used.
    """
    system_message = {"role": "system", "content": instruction}
    messages = [system_message] + prompt

    if debug:
        print(f"\n {prompt =}")
        print(f"messages before the prompting llm: {messages} \n----------------\n ")
        print("\n\n")

    response = send_message_to_openai(model, messages, temperature, max_tokens)
    print(f"\n {response =}")
    text = response["choices"][0]["message"]["content"]
    n_tokens = response["usage"]

    return text, n_tokens


def answer_question_streaming(
    model: str = "gpt-4",
    instruction: str = "",
    prompt: List[Dict[str, str]] = None,
    temperature: float = 0.3,
    max_tokens: int = 1000,
    debug: bool = False,
    callback: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, str]], int, int, int]:
    """
    Obtains an answer to a question from OpenAI's model with streaming.

    Parameters:
        model (str): The model to use. Defaults to "gpt-4".
        instruction (str): Instruction for the model.
        prompt (List[Dict[str, str]]): The prompt for the model as a list of messages.
        temperature (float): Sampling temperature. Defaults to 0.3.
        max_tokens (int): Maximum tokens for the response. Defaults to 1000.
        debug (bool): Whether to debug. Defaults to False.
        callback (Optional[Callable[[str], None]]): Callback function for streaming. Defaults to None.

    Returns:
        Tuple: The updated messages, number of prompt tokens, number of completion tokens, and the total tokens.
    """
    system_message = {"role": "system", "content": instruction}
    messages = [system_message] + (prompt if prompt is not None else [])

    if debug:
        print(f"Debug: Messages before sending to OpenAI: {messages}")

    try:
        response_gen = send_message_to_openai(model, messages, temperature, max_tokens, stream=True)
        output_text = ""
        completion_tokens = 0  # Initialize completion tokens counter

        for chunk in response_gen:
            if "content" in chunk.get("choices", [{}])[0].get("delta", {}):
                r_text = chunk["choices"][0]["delta"]["content"]
                output_text += r_text
                completion_tokens += 1  # Count the packet as one token

                if callback:
                    callback(r_text)

        # Add the final assistant message
        system_message = {"role": "assistant", "content": output_text}
        messages.append(system_message)

        # Calculate prompt tokens
        prompt_tokens = count_tokens(messages[:-1])  # Exclude the last message which is the assistant's reply
        total_tokens = prompt_tokens + completion_tokens

        if debug:
            print(f"Debug: Messages after receiving from OpenAI: {messages}")
            print(f"Debug: Total tokens = {total_tokens}, Prompt tokens = {prompt_tokens}, Completion tokens = {completion_tokens}")

        return messages, prompt_tokens, completion_tokens, total_tokens, output_text

    except Exception as e:
        print(f"Error: {e}")
        return [], 0, 0, 0
