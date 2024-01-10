# pylint: disable=no-name-in-module
from pathlib import Path

from botmerger import SingleTurnContext
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

from copilot.request_condenser import request_condenser
from copilot.specific_repo import REPO_PATH_IN_QUESTION
from copilot.utils.misc import (
    reliable_chat_completion,
    langchain_messages_to_openai,
    FAST_LONG_GPT_MODEL,
    bot_merger,
)

FILE_SNIPPETS_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """\
You are a "copy-paster" and you CANNOT write your own code, but you are good at repeating contents of a file \
that is given to you back to the user verbatim, without any alterations.

Here is the content of `{file_path}`, a file from the `{repo_name}` repo:\
"""
        ),
        HumanMessagePromptTemplate.from_template("{file_content}"),
        SystemMessagePromptTemplate.from_template("And here is a request from the user."),
        HumanMessagePromptTemplate.from_template("USER'S REQUEST: {user_request}"),
        SystemMessagePromptTemplate.from_template(
            """\
Repeat the content of the file `{file_path}` back verbatim, but omit parts that are fully irrelevant to the user's \
request. It's ok if the parts that you do repeat back do not satisfy the user's request fully, though. It's not \
your job to satisfy the user. There is another assistant for that. Your job is only to repeat parts of \
the file that are somewhat relevant. Your response will be used by that other assistant and that other \
assistant will make sure to fulfil the user's request instead of you, so don't worry.

If nothing in this file is relevant to the user's request at all then output just one word: NONE\
"""
        ),
    ]
)


@bot_merger.create_bot
async def extract_snippets(context: SingleTurnContext) -> None:
    file = Path(context.concluding_request.extra_fields["file"])
    request = context.concluding_request.original_message

    standalone_request = await request_condenser.bot.get_final_response(request)

    messages = FILE_SNIPPETS_PROMPT.format_messages(
        repo_name=REPO_PATH_IN_QUESTION.name,
        file_path=file,
        file_content=(REPO_PATH_IN_QUESTION / file).read_text(encoding="utf-8"),
        user_request=standalone_request.content,
    )
    messages = langchain_messages_to_openai(messages)

    completion = await reliable_chat_completion(
        model=FAST_LONG_GPT_MODEL,
        temperature=0,
        pl_tags=["extract_snippets"],
        messages=messages,
    )
    await context.yield_final_response(f"FILE: {file.as_posix()}\n\n{completion}")
