# pylint: disable=broad-exception-caught
import json
import traceback
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from copilot.specific_repo import REPO_PATH_IN_QUESTION, list_files_in_specific_repo
from copilot.utils.cached_completions import RepoCompletions
from copilot.utils.misc import (
    langchain_messages_to_openai,
    FAST_GPT_MODEL,
    FAST_LONG_GPT_MODEL,
    SLOW_GPT_MODEL,
    EMBEDDING_MODEL,
)

EXPLAIN_FILE_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Here is the content of `{file_path}`, a file from the `{repo_name}` repo:"
        ),
        HumanMessagePromptTemplate.from_template("{file_content}"),
        SystemMessagePromptTemplate.from_template("Please explain the content of this file in plain English."),
    ]
)

gpt3_explainer = RepoCompletions(
    repo=REPO_PATH_IN_QUESTION,
    completion_name="gpt3-expl",
    model=FAST_GPT_MODEL,
)
gpt3_long_explainer = RepoCompletions(
    repo=REPO_PATH_IN_QUESTION,
    completion_name="gpt3-long-expl",
    model=FAST_LONG_GPT_MODEL,
)
gpt4_explainer = RepoCompletions(
    repo=REPO_PATH_IN_QUESTION,
    completion_name="gpt4-expl",
    model=SLOW_GPT_MODEL,
)
ada_embedder = RepoCompletions(
    repo=REPO_PATH_IN_QUESTION,
    completion_name="ada2-expl",
    model=EMBEDDING_MODEL,
)


async def explain_repo_file_in_isolation(file: Path | str, gpt4: bool = False) -> str:
    if not isinstance(file, Path):
        file = Path(file)

    messages = EXPLAIN_FILE_PROMPT.format_messages(
        repo_name=REPO_PATH_IN_QUESTION.name,
        file_path=file,
        file_content=(REPO_PATH_IN_QUESTION / file).read_text(encoding="utf-8"),
    )
    messages = langchain_messages_to_openai(messages)

    if gpt4:
        explanation = await gpt4_explainer.file_related_chat_completion(messages=messages, repo_file=file)
    else:
        explanation = await gpt3_explainer.file_related_chat_completion(
            messages=messages, repo_file=file, cache_only=True
        )
        if explanation is None:
            # short model cache miss - try the long model cache
            explanation = await gpt3_long_explainer.file_related_chat_completion(
                messages=messages, repo_file=file, cache_only=True
            )

        if explanation is None:
            # cache miss for both models - generate a new completion
            try:
                explanation = await gpt3_explainer.file_related_chat_completion(messages=messages, repo_file=file)
            except Exception:  # TODO use more specific exceptions ?
                # short model failed - try the long model
                explanation = await gpt3_long_explainer.file_related_chat_completion(messages=messages, repo_file=file)

    return f"FILE: {file.as_posix()}\n\n{explanation}"


async def explain_everything() -> None:
    # repo_files = list_files_in_specific_repo_chunked(reduced_list=True)[0]
    repo_files = list_files_in_specific_repo(reduced_list=True)
    print()
    for file in repo_files:
        print(file)
    print()
    print(len(repo_files))
    print()

    failed_files = []
    for idx, file in enumerate(repo_files):
        try:
            print(idx + 1, "/", len(repo_files), "-", file)
            messages = EXPLAIN_FILE_PROMPT.format_messages(
                repo_name=REPO_PATH_IN_QUESTION.name,
                file_path=file,
                file_content=(REPO_PATH_IN_QUESTION / file).read_text(encoding="utf-8"),
            )
            messages = langchain_messages_to_openai(messages)
            await gpt3_explainer.file_related_chat_completion(messages=messages, repo_file=file)
        except Exception:
            traceback.print_exc()
            failed_files.append(file)

    if failed_files:
        print()
        print("FAILED FILES:")
        print()
        for file in failed_files:
            print(file)
        print()
        print(len(failed_files))
        print()

        files_that_failed_again = []
        for idx, file in enumerate(failed_files):
            try:
                print(idx + 1, "/", len(failed_files), "-", file)
                messages = EXPLAIN_FILE_PROMPT.format_messages(
                    repo_name=REPO_PATH_IN_QUESTION.name,
                    file_path=file,
                    file_content=(REPO_PATH_IN_QUESTION / file).read_text(encoding="utf-8"),
                )
                messages = langchain_messages_to_openai(messages)
                await gpt3_long_explainer.file_related_chat_completion(messages=messages, repo_file=file)
            except Exception:
                traceback.print_exc()
                files_that_failed_again.append(file)

        if files_that_failed_again:
            print()
            print("FAILED AGAIN FILES:")
            print()
            for file in files_that_failed_again:
                print(file)
            print()
            print(len(files_that_failed_again))
            print()

    print("DONE")


async def print_explanation(explainer: RepoCompletions, messages: Iterable[dict[str, str]], file: Path | str) -> None:
    print("====================================================================================================")
    print()
    print(await explainer.file_related_chat_completion(messages=messages, repo_file=file))
    print()
    print(explainer.model)
    print()


async def embed_everything() -> None:
    repo_files = list_files_in_specific_repo(reduced_list=True)
    print()
    for file in repo_files:
        print(file)
    print()
    print(len(repo_files))
    print()

    embeddings = []
    embedded_files = []

    failed_files = []
    for idx, file in enumerate(repo_files):
        try:
            print(idx + 1, "/", len(repo_files), "-", file)
            embeddings.append(
                await ada_embedder.file_related_embedding(await explain_repo_file_in_isolation(file), repo_file=file)
            )
            embedded_files.append(file.as_posix())
        except Exception:
            traceback.print_exc()
            failed_files.append(file)

    if failed_files:
        print()
        print("FAILED FILES:")
        print()
        for file in failed_files:
            print(file)
        print()
        print(len(failed_files))
        print()

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings, dtype=np.float32))  # pylint: disable=no-value-for-parameter
    faiss.write_index(index, str(REPO_PATH_IN_QUESTION / "explanations.faiss"))
    (REPO_PATH_IN_QUESTION / "explanation_files.json").write_text(
        json.dumps(embedded_files, indent=2), encoding="utf-8"
    )
