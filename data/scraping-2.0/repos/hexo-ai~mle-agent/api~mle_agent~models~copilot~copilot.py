"""
The Objective of this script is to build an infernece pipeline which can do the following:
7. Write the end to end logic involved as a function or class (NOT DONE)
8. Wrap this whole thing into a Flask application (NOT DONE)
9. Replace with OPENAI credits keys (NOT DONE)
10. Replace conda environment. Check if  the following code works (NOT DONE)
# Concatenate the dataframes and assign to corpus_df
corpus_df = pd.concat([corpus_df, chunks_with_embeddings_df, chunks_without_embeddings_df], ignore_index=True)
11. Fix long documents issue (HACKILY DONE)
12. Add import statements from the file (NOT DONE)
13. Integrate to a vector store (NOT DONE)
"""
import asyncio
import glob
import json
import textwrap
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, cast
from urllib.parse import urlparse

import aiofiles
import aiofiles.os
import numpy as np
import pandas as pd
import tiktoken
from git import GitCommandError, Repo
from openai.types.chat import ChatCompletionMessageParam
from langchain.text_splitter import MarkdownHeaderTextSplitter

from ...helpers import log
from ...libs.openai import openai_client as openai
from .scripts.code_parser import TreeSitterPythonParser

from dotenv import load_dotenv

load_dotenv()

FloatNDArray = np.ndarray[Any, np.dtype[np.float16]]


async def try_clone_repository(
    repo_url: str,
    *,
    repo_path: Path,
    branch: str,
):
    try:
        await log.ainfo(f"Cloning {( repo_url, branch ) = } to {repo_path = }")
        await asyncio.to_thread(
            Repo.clone_from,
            repo_url,
            repo_path,
            branch=branch,
            depth=1,
        )
        return True
    except GitCommandError as e:
        await log.awarning(f"Failed to clone {branch = }: {e}")
    return False


async def shallow_clone_repository(repo_url: str, repo_path: Path, branch: str | None):
    if await aiofiles.os.path.exists(repo_path):
        return

    branches = ("master", "main")

    if branch:
        branches = (branch, *branches)

    for branch in branches:
        if await try_clone_repository(
            repo_url,
            repo_path=repo_path,
            branch=branch,
        ):
            break
    else:
        raise ValueError(f"None of these {branches=} could be found in the repository.")


async def read_and_chunk_python_document(document_path: str, sem: asyncio.Semaphore):
    async with sem:
        async with aiofiles.open(document_path, "r", encoding="utf-8") as file:
            try:
                document = await file.read()
            except UnicodeDecodeError:
                document = ""

    parser = TreeSitterPythonParser(document=document)
    [chunks, main_code, import_statements] = await asyncio.gather(
        asyncio.to_thread(parser.create_chunks),
        asyncio.to_thread(parser.extract_main_code),  # type: ignore
        asyncio.to_thread(parser.extract_import_statements),  # type: ignore
    )
    chunks_df = pd.DataFrame(chunks)
    new_rows = pd.DataFrame(
        [
            {"code": main_code, "type": "main_code"},
            {"code": import_statements, "type": "imports"},
        ]
    )
    chunks_df = pd.concat([chunks_df, new_rows], ignore_index=True)
    chunks_df = chunks_df[chunks_df["code"].apply(lambda x: len(x)) != 0]
    chunks_df["file_path"] = document_path
    return chunks_df


async def read_and_chunk_markdown_document(document_path: str, sem: asyncio.Semaphore):
    async with sem:
        async with aiofiles.open(document_path, "r", encoding="utf-8") as file:
            try:
                document = await file.read()
            except UnicodeDecodeError:
                document = ""

    headers_to_split_on = [(str("##"), str("Header 2"))]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, return_each_line=False
    )
    md_header_splits = markdown_splitter.split_text(document)
    chunks = []
    for item_num in range(len(md_header_splits)):
        joined_content = " ".join(md_header_splits[item_num].metadata.values())
        joined_content = "\n" + md_header_splits[item_num].page_content
        chunks.append({"code": f"{joined_content}", "type": "markdown"})
    chunks_df = pd.DataFrame(chunks)
    if len(chunks_df) > 0:
        chunks_df = chunks_df[chunks_df["code"].apply(lambda x: len(x)) != 0]
    return chunks_df


async def read_and_chunk_all_python_files(directory_path: Path):
    python_files = glob.glob(f"{directory_path}/**/*.py", recursive=True)
    sem = asyncio.Semaphore(100)
    chunks_df_ls = await asyncio.gather(
        *[read_and_chunk_python_document(file, sem) for file in python_files],
    )
    all_chunks_df: pd.DataFrame = pd.DataFrame()
    all_chunks_df = cast(pd.DataFrame, pd.concat(chunks_df_ls, ignore_index=True))
    return all_chunks_df


async def read_and_chunk_all_markdown_files(directory_path):
    all_chunks_df = pd.DataFrame()

    markdown_files_md = glob.glob(f"{directory_path}/**/*.md", recursive=True)
    markdown_files_mdx = glob.glob(f"{directory_path}/**/*.mdx", recursive=True)
    sem = asyncio.Semaphore(100)
    markdown_files = markdown_files_md + markdown_files_mdx
    chunks_df_ls = await asyncio.gather(
        *[read_and_chunk_markdown_document(file, sem) for file in markdown_files],
    )
    all_chunks_df: pd.DataFrame = pd.DataFrame()
    all_chunks_df = cast(pd.DataFrame, pd.concat(chunks_df_ls, ignore_index=True))

    return all_chunks_df


async def num_tokens(text: str, model: str) -> int:
    """Return the number of tokens in a string."""
    encoding = await asyncio.to_thread(tiktoken.encoding_for_model, model)
    token_ls = await asyncio.to_thread(encoding.encode, text, disallowed_special=())
    num_tokens = len(token_ls)
    return num_tokens


def parse_embedding(embedding: str | list[str]):
    if isinstance(embedding, str):
        return json.loads(embedding)
    elif isinstance(embedding, list):
        return embedding
    else:
        raise ValueError(f"Unexpected value: {embedding}")


async def create_openai_embedding(
    batch: list[str],
    embedding_model_name: str,
) -> list[list[float]]:
    batch = [
        item[:1000]
        if await num_tokens(
            item,
            embedding_model_name,
        )
        > 8192
        else item
        for item in batch
    ]

    response = await openai.embeddings.create(model=embedding_model_name, input=batch)
    await log.ainfo("response", response=len(response.data))
    for i, be in enumerate(response.data):
        assert i == be.index  # double check embeddings are in same order as input

    return [e.embedding for e in response.data]


async def create_openai_embeddings(
    batches: list[list[str]],
    embedding_model_name: str,
) -> AsyncGenerator[tuple[list[list[float]], Literal["pending", "done"]], None]:
    # the total number of tokens in a request must be less than 1M
    # to not hit the rate limit

    embeddings_ls: list[list[float]] = []

    for batch in batches:
        embeddings_ls.extend(
            await create_openai_embedding(
                batch,
                embedding_model_name,
            )
        )
        yield embeddings_ls, "pending"

    yield embeddings_ls, "done"


async def create_query_embedding(query: str, embedding_model_name: str):
    embeddings: list[float] = []
    async for em, status in create_openai_embeddings([[query]], embedding_model_name):
        if status == "done":
            embeddings = list(em[0])
            break
    query_embedding: FloatNDArray = np.array(embeddings).astype(float).reshape(1, -1)
    return query_embedding


def compute_cosine_similarity(
    chunks_embeddings: FloatNDArray, query_embedding: FloatNDArray
):
    log.info("embeddings", embeddings=chunks_embeddings)
    chunk_norms: FloatNDArray = np.linalg.norm(chunks_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    # Compute cosine similarities
    similarities: FloatNDArray = np.dot(chunks_embeddings, query_embedding.T) / (
        chunk_norms[:, np.newaxis] * query_norm
    )
    return similarities


def create_message(
    query: str, messages: list[ChatCompletionMessageParam], top_chunks: list[str]
):
    context = "\n".join(top_chunks)
    content = f"Respond to the query based on the provided context.\
                If the query involves writing code, keep the code concise. \
                Write code only for what the user has asked for \
                Query: {query} \n Context: {context} \n"

    for message in reversed(messages):
        if message["role"] == "user":
            message["content"] = content
            break

    return messages


def get_top_chunks(
    chunks: list[str],
    chunk_embeddings: FloatNDArray,
    query_embedding: FloatNDArray,
    top_n: int,
) -> tuple[list[str], list[FloatNDArray]]:
    similarities = compute_cosine_similarity(chunk_embeddings, query_embedding)
    top_indices = np.argsort(similarities.flatten())[-top_n:][
        ::-1
    ]  # Reverse the indices to get them in descending order
    top_chunks: list[str] = [chunks[i] for i in top_indices]
    top_similarities = [similarities.flatten()[i] for i in top_indices]
    return top_chunks, top_similarities


def add_imports_to_code(imports: list[str], code: str):
    code = textwrap.dedent(code)
    import_str = "\n".join(imports)
    import_str = textwrap.dedent(import_str)
    return import_str + "\n" + code


async def ask_gpt(
    query: str,
    messages: list[ChatCompletionMessageParam],
    top_chunks: list[str],
    model: str,
    repo_url: str,
    similarity_scores: list[FloatNDArray],
    trace_id: str,
):
    messages = create_message(query, messages, top_chunks=top_chunks)
    response = await openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=True,
        metadata={
            "repo_url": repo_url,
            "query": query,
            "chunks": {
                "top_chunks": top_chunks,
                "similarity_scores": similarity_scores,
            },
        },
        trace_id=trace_id,
    )  # type: ignore
    async for chunk in response:
        yield chunk


class InferencePipeline:
    def __init__(
        self,
        repo_url: str,
        *,
        repo_parent_path: str | None = None,
        start_index_folder_path: str = "",
        branch: str | None = None,
    ):
        self.repo_url = repo_url
        self.repo_parent_path = (
            Path(repo_parent_path) if repo_parent_path else Path.cwd() / "repos"
        )
        self.branch = branch
        self.start_index_folder_path = Path(start_index_folder_path)
        columns = [
            "repo_url",
            "file_path",
            "code",
            "start_line_num",
            "end_line_num",
            "type",
            "parser_type",
            "embedding",
        ]
        self.corpus_df = pd.DataFrame(columns=columns)

    async def clone_and_process_repo(self):
        repo_path = Path.joinpath(
            self.repo_parent_path,
            *urlparse(self.repo_url).path.split(".")[0].split("/")[1:3],
        )
        repo_embedding_path = repo_path / "embeddings.csv"
        if await aiofiles.os.path.exists(repo_embedding_path):
            self.corpus_df = pd.read_csv(repo_embedding_path)
            yield "Loaded repo...\n\n"
            yield ""
            return

        await log.ainfo("corpus_df", corpus_df=self.corpus_df)

        yield "Loading and processing repo...\n\n"

        await shallow_clone_repository(
            repo_url=self.repo_url,
            repo_path=repo_path,
            branch=self.branch,
        )

        yield "Cloned repo...\n\n"

        await asyncio.sleep(0.5)

        yield "Processing repo...\n\n"

        all_python_chunks_df = await read_and_chunk_all_python_files(
            directory_path=repo_path
        )
        all_markdown_chunks_df = await read_and_chunk_all_markdown_files(
            directory_path=repo_path
        )

        all_chunks_df = pd.concat([all_python_chunks_df, all_markdown_chunks_df])

        yield "Chunked repo...\n\n"

        await log.ainfo("all_chunks_df", all_chunks_df=all_python_chunks_df)

        chosen_types = ["class_definition", "function_definition", "markdown"]

        chunks_with_embeddings_df = all_chunks_df[
            all_chunks_df["type"].isin(chosen_types)
        ]
        chunks_without_embeddings_df = all_chunks_df[
            ~all_chunks_df["type"].isin(chosen_types)
        ]

        yield "Creating embeddings...\n\n"

        batch_size = 300
        code_inputs = chunks_with_embeddings_df["code"].tolist()
        batches: list[list[str]] = [
            code_inputs[batch_start : batch_start + batch_size]
            for batch_start in range(0, len(code_inputs), batch_size)
        ]
        no_of_batches = len(batches)
        await log.ainfo(
            "batches", batches=[len(batch) for batch in batches], len=len(batches)
        )

        async for em, status in create_openai_embeddings(
            batches,
            "text-embedding-ada-002",
        ):
            if status == "done":
                chunks_with_embeddings_df.loc[:, "embedding"] = em
                break
            yield f"Embedding: Batch {int(round(len(em)/batch_size, batch_size))} / {no_of_batches} in progress...\n\n"

        yield "Created embeddings...\n\n"

        # indicate end of repo processing
        yield ""

        self.corpus_df = pd.concat(
            [
                df
                for df in [
                    self.corpus_df,
                    chunks_with_embeddings_df,
                    chunks_without_embeddings_df,
                ]
                if not df.empty
            ],
            ignore_index=True,
        )
        self.corpus_df.to_csv(repo_embedding_path, index=False)

    async def get_latest_prompt(self, messages: list[ChatCompletionMessageParam]):
        for message in reversed(messages):
            if message["role"] == "user":
                return str(message["content"])
        return ""

    async def compute_similarities(self, query: str, top_n: int):
        chosen_types = ["class_definition", "function_definition", "markdown"]
        chunks_with_embeddings_df = cast(
            pd.DataFrame, self.corpus_df[self.corpus_df["type"].isin(chosen_types)]
        )
        chunks_with_import_statements = cast(
            pd.DataFrame,
            self.corpus_df[
                self.corpus_df["type"].isin(
                    ["import_statement", "import_from_statement"]
                )
            ],
        )

        chunk_embeddings = np.array(
            chunks_with_embeddings_df["embedding"].apply(parse_embedding).tolist()
        ).astype(float)
        query_embedding = await create_query_embedding(query, "text-embedding-ada-002")
        chunks = chunks_with_embeddings_df["code"].tolist()
        top_chunks, top_chunks_similarity_scores = get_top_chunks(
            chunks, chunk_embeddings, query_embedding, top_n=3
        )
        top_chunk_with_imports = add_imports_to_code(
            imports=chunks_with_import_statements["code"].tolist(), code=top_chunks[0]
        )
        return top_chunks, top_chunks_similarity_scores, top_chunk_with_imports

    async def get_response(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
        user_id: str,
        top_n=3,
    ):
        user_latest_prompt = await self.get_latest_prompt(messages)
        (
            top_chunks,
            top_chunks_similarity_scores,
            top_chunk_with_imports,
        ) = await self.compute_similarities(query=user_latest_prompt, top_n=top_n)
        async for sample_response in ask_gpt(
            user_latest_prompt,
            messages,
            top_chunks=top_chunks[:2],
            model=model,
            repo_url=self.repo_url,
            similarity_scores=top_chunks_similarity_scores[:2],
            trace_id=user_id,
        ):
            await asyncio.sleep(0.01)
            str_resp = sample_response.model_dump_json() + "\n"
            await log.ainfo("str_resp", str_resp=str_resp)
            yield str_resp


if __name__ == "__main__":
    pipeline = InferencePipeline(
        repo_url="https://github.com/langchain-ai/langchain.git",
        repo_parent_path="samples",
        start_index_folder_path="langchain/libs/langchain/langchain/document_transformers",
    )
    pipeline.clone_and_process_repo()
    query = "Write python function to read a HTML file and transform it into text using Langchain"
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are an expert programmer",
        },
        {"role": "user", "content": query},
    ]
    model = "gpt-3.5-turbo"
    pipeline.get_response(messages=messages, model=model, user_id="testing")
