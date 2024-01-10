import os
import openai
from typing import Iterator, Optional
from .context import Context
from .history import History


class CommandMissingException(Exception):
    pass


def validate_api_key(openai_api_key) -> bool:
    try:
        openai.Model.list(api_key=openai_api_key)
        return True
    except Exception:
        return False


def translate_with_history(
    instruction: str,
    history_path: Optional[os.PathLike],
    context: Context,
    openai_api_key: str,
) -> Iterator[str]:
    history: History = History.create_local()
    if history_path:
        history = History.load_from_file(history_path)
    history.append_user_message(instruction)

    commands_chunks = []
    chunks = request_command(history, context, openai_api_key)
    try:
        for chunk in filter_assistant_message(chunks):
            yield chunk
            commands_chunks.append(chunk)
    except CommandMissingException:
        corrective_history = history.copy()
        corrective_history.append_user_message("stick to the format")
        chunks = request_command(corrective_history, context, openai_api_key)
        for chunk in filter_assistant_message(chunks):
            yield chunk
            commands_chunks.append(chunk)

    command = "".join(commands_chunks)
    history.append_assistant_message(command)
    history.checkpoint()


def filter_assistant_message(chunks: Iterator[str]) -> Iterator[str]:
    # Filter all the chunks between COMMAND and END
    try:
        while "COMMAND" not in next(chunks):
            pass

        first_chunk = next(chunks)
        yield first_chunk[1:]  # removes the space after "COMMAND"

        while "END" not in (chunk := next(chunks)):
            yield chunk
    except StopIteration:
        raise CommandMissingException
    return


def request_command(
    current_history: History, context: Context, openai_api_key: str
) -> Iterator[str]:
    openai.api_key = openai_api_key

    system_prompt = system_prompt_from_context(context)
    system_message = {"role": "system", "content": system_prompt}
    chat_messages = current_history.get_messages()
    messages = [system_message] + chat_messages

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0,
        stream=True,
        messages=messages,
    )

    for chunk in response:
        if "content" in chunk.choices[0].delta:
            yield chunk.choices[0].delta.content


def system_prompt_from_context(context: Context) -> str:
    return f"""
    You are a CLI tool called comai  with access to a {context.shell} shell on {context.system}. Your goal is to translate the instruction from the user into a command. 

    ALWAYS use the following format, with no additional comments, explanation, or notes before or after AT ALL:

    COMMAND <the command> END

    Example:

    User:
    show files

    Your answer:
    COMMAND ls END
    """
