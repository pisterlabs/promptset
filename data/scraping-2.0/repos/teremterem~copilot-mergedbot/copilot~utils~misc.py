from pathlib import Path
from typing import Iterable

from botmerger import MergedMessage, MergedParticipant, YamlLogBotMerger
from langchain.chat_models.openai import _convert_message_to_dict
from langchain.schema import BaseMessage

FAST_GPT_MODEL = "gpt-3.5-turbo-0613"
# TODO revert back to more powerful models
FAST_LONG_GPT_MODEL = FAST_GPT_MODEL
SLOW_GPT_MODEL = FAST_GPT_MODEL
# FAST_LONG_GPT_MODEL = "gpt-3.5-turbo-16k-0613"
# SLOW_GPT_MODEL = "gpt-4-0613"
EMBEDDING_MODEL = "text-embedding-ada-002"

CHAT_HISTORY_MAX_LENGTH = 20

bot_merger = YamlLogBotMerger(Path(__file__).parents[2] / "merged_log.yaml", serialization_enabled=False)


def get_openai_role_name(message: MergedMessage, this_bot: MergedParticipant) -> str:
    return "assistant" if message.sender == this_bot else "user"


def langchain_messages_to_openai(messages: Iterable[BaseMessage]) -> list[dict[str, str]]:
    return [_convert_message_to_dict(msg) for msg in messages]


def sort_paths(paths: Iterable[Path], case_insensitive: bool = False) -> list[Path]:
    return sorted(paths, key=lambda p: (p.as_posix().lower(), p.as_posix()) if case_insensitive else p.as_posix())


async def reliable_chat_completion(**kwargs) -> str:
    # pylint: disable=import-outside-toplevel,no-name-in-module
    from promptlayer import openai

    response = await openai.ChatCompletion.acreate(**kwargs)
    completion = response.choices[0]
    if completion.finish_reason != "stop":
        raise RuntimeError(f"Incomplete chat completion (finish_reason: {completion.finish_reason})")
    return completion.message.content


def format_conversation_for_single_message(conversation: Iterable[MergedMessage], this_bot: MergedParticipant) -> str:
    conversation_str = "\n\n".join(
        f"" f"{get_openai_role_name(msg, this_bot).upper()}: {msg.content}" for msg in conversation
    )
    return conversation_str
