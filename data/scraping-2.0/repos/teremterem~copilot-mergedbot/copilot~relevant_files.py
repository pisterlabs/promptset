# pylint: disable=no-name-in-module
import itertools
import json
import re

import faiss
import numpy as np
from botmerger import SingleTurnContext
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage
from promptlayer import openai

from copilot.explain_repo import explain_repo_file_in_isolation
from copilot.request_condenser import request_condenser
from copilot.specific_repo import REPO_PATH_IN_QUESTION
from copilot.utils.misc import (
    EMBEDDING_MODEL,
    langchain_messages_to_openai,
    reliable_chat_completion,
    FAST_GPT_MODEL,
    bot_merger,
)

RELEVANT_FILES_PROMPT_PREFIX = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """\
You are an AI assistant that is good at answering questions about the concepts that can be found in the repository \
by the name `{repo_name}`.

Below are the summaries of some source code files from `{repo_name}` repo which may or may not be relevant to the \
conversation that you are currently having with the user. Each file has a number assigned to it in square brackets.\
"""
        ),
    ]
)
RELEVANT_FILES_PROMPT_SUFFIX = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("And here is a request from the user."),
        HumanMessagePromptTemplate.from_template("USER'S REQUEST: {user_request}"),
        SystemMessagePromptTemplate.from_template(
            """\
Please select the numbers which correspond to the files that are at least vaguely relevant to the user's request. \
DO NOT EXPLAIN ANYTHING, JUST LIST THE NUMBERS. DO NOT EVEN CITE THE FILE PATHS THEMSELVES, YOUR OUTPUT SHOULD BE \
NUMBERS ONLY.\
"""
        ),
    ]
)

EXPLANATIONS_FAISS = faiss.read_index(str(REPO_PATH_IN_QUESTION / "explanations.faiss"))
INDEXED_EXPL_FILES = json.loads((REPO_PATH_IN_QUESTION / "explanation_files.json").read_text(encoding="utf-8"))


@bot_merger.create_bot
async def relevant_files(context: SingleTurnContext) -> None:
    request = context.concluding_request.original_message

    standalone_request = await request_condenser.bot.get_final_response(request)

    result = await openai.Embedding.acreate(
        model=EMBEDDING_MODEL,
        temperature=0,
        pl_tags=["embedding"],
        input=[standalone_request.content],
    )
    embedding = result["data"][0]["embedding"]
    _, indices = EXPLANATIONS_FAISS.search(np.array([embedding], dtype=np.float32), 5)

    fetched_files = [INDEXED_EXPL_FILES[idx] for idx in indices[0]]
    file_explanations = [
        {"file": file, "explanation": await explain_repo_file_in_isolation(file=file)} for file in fetched_files
    ]

    prompt_prefix = RELEVANT_FILES_PROMPT_PREFIX.format_messages(repo_name=REPO_PATH_IN_QUESTION.name)
    recalled_files = [
        HumanMessage(content=f"[{idx}] {file_explanation['explanation']}")
        for idx, file_explanation in enumerate(file_explanations, start=1)
    ]
    prompt_suffix = RELEVANT_FILES_PROMPT_SUFFIX.format_messages(user_request=standalone_request.content)

    prompt_openai = langchain_messages_to_openai(itertools.chain(prompt_prefix, recalled_files, prompt_suffix))

    completion = await reliable_chat_completion(
        model=FAST_GPT_MODEL,
        temperature=0,
        pl_tags=["relevant_files"],
        messages=prompt_openai,
    )
    file_numbers_to_keep = [int(n) for n in re.findall(r"\d+", completion)]

    # TODO make the code below less ugly ?
    num_of_responses = 0
    for i, msg in enumerate(file_explanations, start=1):
        if i > 1 and i not in file_numbers_to_keep:
            # the file is neither the first one in the similarity list nor it was chosen by the LLM explicitly
            continue

        num_of_responses += 1
        if num_of_responses > 1:
            await context.yield_final_response(msg["explanation"], extra_fields={"file": msg["file"]})
            # we are returning no more than 2 files
            return
        await context.yield_interim_response(msg["explanation"], extra_fields={"file": msg["file"]})
