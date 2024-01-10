# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
from typing import List, Dict
from pathlib import Path
import json
import traceback

import openai

from controller.load_openai import load_openai
from controller.create_logger import create_logger
from controller.ai_functions import ai_available_functions
from controller.conversation_handler import truncate_conversation


# Load the OpenAI API key
client = load_openai()

# Create logger
module_logger = create_logger(
    logger_name=__name__,
    logger_filename="completion_create.log",
    log_directory="logs",
    add_date_to_filename=False,
)

# Constants
MODEL_USED = "gpt-3.5-turbo-0613"  # "gpt-4-0613"


def generate_message(role: str, content: str) -> Dict | None:
    """
    Generate a message for the OpenAI API.

    Args:
        role (str): The role of the message either "system", "user", or
        "assistant". content (str): The content of the message.
    returns:
        Dict: The message to send to the OpenAI API. e.g. {"role": "user",
        "content": "Hello!"} if the message is not a valid message then return
        None.
    """
    message = {"role": role, "content": content}
    return message


def get_response(
    messages: List[Dict],
    model: str = "gpt-3.5-turbo-0613",
    temperature: float = 1.0,
    max_tokens: int = 200,
    functions: List[str] = None,
    function_call: str = "none",
) -> str:
    """
    Get the answer from the OpenAI API.

    Args:
        messages (List): The messages to send to the OpenAI API.

    Returns:
        str: The answer from the OpenAI API.
    """
    GPT4_TOKEN_LIMIT = 4097

    # create a copy of the messages list
    new_messages = messages.copy()

    while True:
        try:
            if functions is not None and function_call != "none":
                first_response = client.chat.completions.create(
                    messages=new_messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,  # 8,192 tokens is the max for GPT-4
                    # stop=["\n\n", "Link:", "system:"],
                    functions=functions,
                    function_call=function_call,  # auto is default, but we'll be explicit
                )
            else:
                first_response = client.chat.completions.create(
                    messages=new_messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,  # 8,192 tokens is the max for GPT-4
                    # stop=["\n\n", "Link:", "system:"],
                )
            module_logger.info(
                "Before processing: first_response = %s", repr(first_response)
            )
        except openai.APITimeoutError as error:
            # Handle timeout error, e.g. retry or log
            module_logger.critical(
                "openai.APITimeoutError:\nOpenAI API request timed out: %s\nFull traceback:\n%s",
                error,
                traceback.format_exc(),
            )
            return error
        except openai.APIConnectionError as error:
            # Handle authentication error, e.g. check credentials or log
            module_logger.critical(
                (
                    "openai.APIConnectionError:\n"
                    "OpenAI API request was not authorized: %s\n"
                    "Full traceback:\n%s"
                ),
                error,
                traceback.format_exc(),
            )
            return error
        except openai.APIResponseValidationError as error:
            # Handle permission error, e.g. check scope or log
            module_logger.critical(
                (
                    "openai.APIResponseValidationError:\n"
                    "OpenAI API request was not permitted: %s\n"
                    "Full traceback:\n%s"
                ),
                error,
                traceback.format_exc(),
            )
            return error
        except openai.APIStatusError as error:
            # Handle status error, e.g. retry or log
            module_logger.critical(
                (
                    "openai.APIStatusError:\n"
                    "OpenAI API request got a status: %s\n"
                    "Full traceback:\n%s"
                ),
                error,
                traceback.format_exc(),
            )
            return error
        except openai.APIError as error:
            # Handle API error, e.g. retry or log
            module_logger.critical(
                "openai.APIError:\nOpenAI API returned an API Error: %s", error
            )
            return error
        else:
            module_logger.error("Error ==> %s", first_response.choices[0].message.content)

            # Get the response message
            response_message = first_response.choices[0].message

            # Check if GPT wanted to call a function
            if response_message.get("function_call"):
                module_logger.info(
                    "Function call: %s", response_message["function_call"]
                )

                # Call the function
                # Note: the JSON response may not always be valid; be sure to handle errors
                function_name = response_message["function_call"]["name"]
                fuction_to_call = ai_available_functions[function_name]
                function_args = json.loads(
                    response_message["function_call"]["arguments"]
                )
                function_response = fuction_to_call(**function_args)

                module_logger.info(
                    "Before Send the info on the function call and function response to GPT response_message = %s",
                    repr(response_message),
                )

                # Send the info on the function call and function response to GPT
                # extend conversation with assistant's reply
                new_messages.append(response_message)

                # convert function response to unicode
                function_response_unicode = bytes(
                    function_response,
                    "utf-8",
                ).decode("unicode-escape")

                # Create a new message with the function response
                function_response_dict = {
                    "role": "function",
                    "name": function_name,
                    "content": function_response_unicode,
                }

                # extend conversation with function response
                new_messages.append(function_response_dict)
                messages.append(function_response_dict)

                second_response = client.chat.completions.create(
                    model=model,  # "gpt-4-0613", "gpt-3.5-turbo-0613",
                    messages=new_messages,
                )  # get a new response from GPT where it can see the function response
                module_logger.info("second_response = %s", repr(second_response))

                return second_response.choices[0].message.content
            else:
                return first_response.choices[0].message.content
        finally:
            module_logger.info(
                "Finally ==> %s", first_response.choices[0].message.content
            )

def save_conversation(persona: Dict):
    """Save the conversation to the conversation file."""
    # if 'persona' argument is not a dictionary then raise a TypeError
    if not isinstance(persona, dict):
        raise TypeError("persona argument must be a dictionary")

    with open(persona["conversation_file_path"], mode="r+", encoding="utf-8") as file:
        TEMPLATE = f"""\"\"\"This is an example of a conversation that can be used by the script.\"\"\"
from pathlib import Path
from controller.vision.eyes import Eyes


# Constants
DIRECTORY = Path(__file__).parent
FILENAME = Path(__file__).name


# This dictionary is used to save the conversation to a file.
persona = {{
    "name": "{persona["name"]}",
    "age": {persona["age"]},
    "selected_voice": "{persona["selected_voice"]}",
    "target_language": "{persona["target_language"]}",
    "conversation_file_path": DIRECTORY / FILENAME,
    "audio_output_path": Path('{persona["audio_output_path"].as_posix()}'),
    "elevenlabs_voice_model": "{persona["elevenlabs_voice_model"]}",
    "gpt_model": "{persona["gpt_model"]}",
    "tools": {persona["tools"]},
    "available_functions": {{
        "take_picture_and_process": Eyes.take_picture_and_process,
    }},  # only one function in this example, but you can have multiple
    "tool_choice": "auto",  # auto is default, but we'll be explicit
}}

# Add system to describe the persona.
persona[
    "system"
] = f\"\"\"You are an artificial intelligence powered friend.
Your name is {{persona["name"]}}, you are {{persona["age"]}} years old, you speak in {{persona["target_language"]}} only.
You are talking with Jorge, inside VRChat.
\"\"\"

# Add system to messages.
persona["messages"] = {persona["messages"]}

persona["old_messages"] = {persona["old_messages"]}

"""
        with open(persona["conversation_file_path"], mode="r+", encoding="utf-8") as file:
            file.seek(0)
            file.write(TEMPLATE)
            file.truncate()


def main():
    # # Loop until the user says "bye"
    # while True:
    #     # Prompt the user for input
    #     user_input = input("\nUser: ")

    #     # Save the user input to the messages list
    #     MESSAGES.append(generate_message("user", user_input))

    #     response = get_answer(MESSAGES)["choices"][0]["message"]["content"]
    #     print(f"\nAssistant: {response}")

    #     # Save the response to the messages list
    #     MESSAGES.append(generate_message("assistant", response))

    #     # If the user says "bye" then break out of the loop
    #     if user_input == "bye":
    #         break

    # Save the MESSAGES list to the tsundere_ai_conversation.py file.
    # conversation_path = Path(__file__).parent / "conversations" / "tsundere_ai_conversation.py"
    # save_conversation()
    pass


if __name__ == "__main__":
    main()
