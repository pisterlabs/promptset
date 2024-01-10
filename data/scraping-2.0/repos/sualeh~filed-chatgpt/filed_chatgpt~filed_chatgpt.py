"""Main module for filed_chatgpt."""
import argparse

from filed_chatgpt.dialog_turns import DialogTurns
from filed_chatgpt.message import Message
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletion


def chat_loop(dialog_turns: DialogTurns):
    """
    Continuously prompt the user, and print response until the user exits.

    Args:
        dialog_turns (DialogTurns): Instance of DialogTurns for
            managing conversation history.
    """
    while True:
        user_prompt = input("?: ")
        if user_prompt.lower() in ["exit", "quit"]:
            break  # Exit the loop
        dialog_turns.add_message(Message.from_prompt(user_prompt))
        completion: str = __complete(dialog_turns)
        print(completion)
        print()


def __complete(dialog_turns: DialogTurns) -> str:
    """
    Generate a completion for the conversation using the OpenAI API.

    Args:
        dialog_turns (DialogTurns): Instance of DialogTurns containing
            the conversation history.

    Returns:
        str: The AI-generated response to the conversation.
    """
    try:
        client = OpenAI()
        chat_completion: ChatCompletion = client.chat.completions.create(
            model=dialog_turns.model, messages=dialog_turns.messages()
        )
        dialog_turns.add_message(Message.from_completion(chat_completion))
        reply = chat_completion.choices[0].message.content
    except OpenAIError as e:
        reply = str(e)

    return reply


def get_args() -> dict:
    """
    Parse command-line arguments and return them as a dictionary.

    Returns:
        dict: Dictionary containing the defined arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="ChatGPT model",
        default="gpt-3.5-turbo",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="Chat output file in YAML/ Markdown format",
        required=True,
    )

    args = parser.parse_args()

    arg_dict = {"model": args.model, "output_file": args.output_file}
    return arg_dict
