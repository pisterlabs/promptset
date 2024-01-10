# pylint: disable=no-name-in-module
import re

from botmerger import SingleTurnContext
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

from copilot.utils.misc import (
    FAST_GPT_MODEL,
    reliable_chat_completion,
    langchain_messages_to_openai,
    get_openai_role_name,
    bot_merger,
    CHAT_HISTORY_MAX_LENGTH,
)

CHAT_HISTORY_FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Here is a conversation history where each utterance has a number assigned to it (in brackets)."
        ),
        HumanMessagePromptTemplate.from_template("{chat_history}"),
        SystemMessagePromptTemplate.from_template(
            "And here is the current message (the one that goes right after the conversation history)."
        ),
        HumanMessagePromptTemplate.from_template("{current_message}"),
        SystemMessagePromptTemplate.from_template(
            """\
Please select the numbers of the utterances which are important in relation to the current message and need to be \
kept. DO NOT EXPLAIN ANYTHING, JUST LIST THE NUMBERS.\
"""
        ),
    ]
)


@bot_merger.create_bot
async def chat_history_filter(context: SingleTurnContext) -> None:
    request = context.concluding_request.original_message
    assistant_in_question = request.receiver

    history = await request.get_conversation_history(max_length=CHAT_HISTORY_MAX_LENGTH)

    if history:
        chat_history_parts = [
            f"[{i}] {get_openai_role_name(msg, assistant_in_question).upper()}: {msg.content}"
            for i, msg in enumerate(history, start=1)
        ]
        chat_history = "\n\n".join(chat_history_parts)
        current_message = (
            f"[CURRENT] {get_openai_role_name(request, assistant_in_question).upper()}: {request.content}"
        )
        filter_prompt = CHAT_HISTORY_FILTER_PROMPT.format_messages(
            chat_history=chat_history,
            current_message=current_message,
        )
        filter_prompt = langchain_messages_to_openai(filter_prompt)
        message_numbers_to_keep = await reliable_chat_completion(
            model=FAST_GPT_MODEL,
            temperature=0,
            pl_tags=["chat_history_filter"],
            messages=filter_prompt,
        )
        message_numbers_to_keep = [int(n) for n in re.findall(r"\d+", message_numbers_to_keep)]
        history = [msg for i, msg in enumerate(history, start=1) if i in message_numbers_to_keep]

    await context.yield_from(history, still_thinking=True)
    await context.yield_final_response(request)
