from pathlib import Path
from typing import AsyncGenerator

from langchain import LLMChain
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from mergedbots import MergedBot, MergedMessage
from mergedbots.ext.discord_integration import DISCORD_MSG_LIMIT

from experiments.common.bot_manager import bot_manager, FAST_GPT_MODEL
from experiments.common.repo_access_utils import list_files_in_repo

# `gpt-3.5-turbo` (unlike `gpt-4`) might pay more attention to `user` messages than it would to `system` messages
EXTRACT_FILE_PATH_PROMPT = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template("{file_list}"),
        HumanMessagePromptTemplate.from_template(
            """\
HERE IS A REQUEST FROM A USER:

{request}"""
        ),
        HumanMessagePromptTemplate.from_template(
            """\
IF THE USER IS ASKING FOR A FILE FROM THE REPO ABOVE, PLEASE RESPOND WITH THE FOLLOWING JSON:
{{
    "file": "path/to/file"
}}

IF THE USER IS ASKING FOR A FILE THAT IS NOT LISTED ABOVE OR THERE IS NO MENTION OF A FILE IN THE USER'S REQUEST, \
PLEASE RESPOND WITH THE FOLLOWING JSON:
{{
    "file": ""  // empty string
}}

YOUR RESPONSE:
{{
    "file": "\
"""
        ),
    ]
)


@bot_manager.create_bot(handle="RepoPathBot")
async def repo_path_bot(bot: MergedBot, message: MergedMessage) -> AsyncGenerator[MergedMessage, None]:
    feedback = await (await bot.manager.find_bot("FeedbackBot")).get_final_response(
        await message.final_bot_response(bot, "hey, say something!")
    )
    yield feedback
    repo_dir = (Path(__file__).parents[3] / "mergedbots").resolve().as_posix()
    yield await message.final_bot_response(bot, repo_dir)


@bot_manager.create_bot(handle="ListRepoTool", description="Lists all the files in a repo.")
async def list_repo_tool(bot: MergedBot, message: MergedMessage) -> AsyncGenerator[MergedMessage, None]:
    repo_dir_msg = None
    async for result in repo_path_bot.bot.fulfill(message):
        repo_dir_msg = result
        yield result

    repo_dir = Path(repo_dir_msg.content)

    file_list = list_files_in_repo(repo_dir)
    file_list_strings = [file.as_posix() for file in file_list]
    file_list_string = "\n".join(file_list_strings)

    result = (
        f"Here is the complete list of files that can be found in `{repo_dir.name}` repo:\n"
        f"```\n"
        f"{file_list_string}\n"
        f"```"
    )
    yield await message.final_bot_response(bot, result, custom_fields={"file_list": file_list_strings})


@bot_manager.create_bot(handle="ReadFileBot", description="Reads a file from the repo.")
async def read_file_bot(bot: MergedBot, message: MergedMessage) -> AsyncGenerator[MergedMessage, None]:
    file_list_msg = await list_repo_tool.bot.get_final_response(message)
    file_set = set(file_list_msg.custom_fields["file_list"])

    chat_llm = PromptLayerChatOpenAI(
        model_name=FAST_GPT_MODEL,
        temperature=0.0,
        model_kwargs={
            "stop": ['"', "\n"],
            "user": str(message.originator.uuid),
        },
        pl_tags=["read_file_bot"],
    )
    llm_chain = LLMChain(
        llm=chat_llm,
        prompt=EXTRACT_FILE_PATH_PROMPT,
    )
    file_path = await llm_chain.arun(request=message.content, file_list=file_list_msg.content)

    if file_path and file_path in file_set:
        yield await message.interim_bot_response(bot, file_path)

        yield await message.final_bot_response(
            bot,
            Path(REPO_DIR, file_path).read_text(encoding="utf-8"),
            custom_fields={"success": True},
        )
    else:
        yield await message.final_bot_response(
            bot,
            f"{file_list_msg.content}\n" f"Please specify the file you want to read.",
            custom_fields={"success": False},
        )


@bot_manager.create_bot(handle="EditFileBot")
async def edit_file_bot(bot: MergedBot, message: MergedMessage) -> AsyncGenerator[MergedMessage, None]:
    read_file_responses = await read_file_bot.bot.list_responses(message)

    file_path_msg = None
    file_content_msg = None
    if read_file_responses[-1].custom_fields.get("success"):
        file_path_msg = read_file_responses[-2]
        file_content_msg = read_file_responses[-1]

    if file_content_msg:
        yield file_path_msg
        yield await message.final_bot_response(bot, f"```\n{file_content_msg.content[:DISCORD_MSG_LIMIT - 10]}\n```")
    else:
        yield read_file_responses[-1]

    chat_llm = PromptLayerChatOpenAI(
        model_name=FAST_GPT_MODEL,
        temperature=0.0,
        model_kwargs={
            "stop": ['"', "\n"],
            "user": str(message.originator.uuid),
        },
        pl_tags=["edit_file_bot"],
    )
    # llm_chain = LLMChain(
    #     llm=chat_llm,
    #     prompt=...,
    # )
