#!python3
import requests
from functools import lru_cache
import pathlib
import subprocess
from pydantic import BaseModel
import tempfile
import pydantic
import openai
import json
from rich.console import Console
from icecream import ic
import typer
import time
import os
from rich import print
from loguru import logger
from pudb import set_trace
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from typing_extensions import Annotated
from openai_wrapper import choose_model, remaining_response_tokens
from fastapi import FastAPI
server = FastAPI()

app = typer.Typer()
console = Console()

chroma_db_dir = "blog.chroma.db"
embeddings = OpenAIEmbeddings()


# TODO: Use UnstructuredMarkdownParser
# Interesting trade off here, if we make chunks bigger we can have more context
# If we make chunk smaller we can inject more chunks
chunk_size = 1024


def chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=0):
    splitter = CharacterTextSplitter(
        separator=" ", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    for document in documents:
        for chunk in splitter.split_text(document.page_content):
            yield Document(page_content=chunk, metadata=document.metadata)


def get_blog_content(path):
    # set_trace()
    repo_path = pathlib.Path(os.path.expanduser(path))

    markdown_files = list(repo_path.glob("*/*.md"))
    for markdown_file in markdown_files:
        with open(markdown_file, "r") as f:
            yield Document(
                page_content=f.read(),
                metadata={"source": str(markdown_file.relative_to(repo_path))},
            )


@app.command()
def build():
    docs = list(get_blog_content("~/blog"))
    # ic(docs[1])
    chunks = list(chunk_documents(docs))
    # ic(chunks[3])
    # set_trace()
    search_index = Chroma.from_documents(
        chunks, embeddings, persist_directory=chroma_db_dir
    )
    search_index.persist()


### Todo, move this to openai_wrapper
def base_query(
    tokens: int = 300,
    debug: bool = False,
    prompt_to_gpt="replace_prompt",
    system_prompt="You are a helpful assistant.",
    u4=False,
):
    model, tokens = choose_model(u4, tokens)

    # Define the messages for the chat
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_to_gpt},
    ]

    output_tokens = remaining_response_tokens(model, system_prompt, prompt_to_gpt)

    if debug:
        ic(system_prompt)
        # ic(prompt_to_gpt)
        ic(model)
        ic(output_tokens)

    start = time.time()
    response_contents = [""]
    first_chunk = True
    for chunk in openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=output_tokens,
        temperature=0.7,
        stream=True,
    ):
        if not "choices" in chunk:
            continue

        for elem in chunk["choices"]:  # type: ignore

            if first_chunk:
                if debug:
                    out = f"First Chunk took: {int((time.time() - start)*1000)} ms"
                    ic(out)
                first_chunk = False
            delta = elem["delta"]
            delta_content = delta.get("content", "")
            response_contents[elem["index"]] += delta_content

    text = ""
    for i, content in enumerate(response_contents):
        text = content

    if debug:
        out = f"All chunks took: {int((time.time() - start)*1000)} ms"
        ic(out)
    return text


@app.command()
def chunk_md(
    path: Annotated[str, typer.Argument()] = "~/blog/_posts/2020-04-01-Igor-Eulogy.md"
):
    from unstructured.partition.md import partition_md

    elements = partition_md(filename=os.path.expanduser(path))
    ic(elements)

class Fact(BaseModel):
    source: str
    content: str
    def to_prompt(self):
        return f"""---(FACT)---
SOURCE FILE PATH:
{self.source}
FACT:
{self.content}
---"""

def facts_to_prompt(facts):
    return "\n".join([f.to_prompt() for f in facts])

@app.command()
def ask(
    question: Annotated[
        str, typer.Argument()
    ] = "What are the roles from Igor's Eulogy, answer in bullet form",
    facts: Annotated[int, typer.Option()] = 5,
    u4: bool = typer.Option(False),
    debug: bool = typer.Option(True),
):

    model = "gpt-4" if u4 else "gpt-3.5-turbo-16k"
    model, max_tokens = choose_model(u4)
    if debug:
        ic(model)
        ic(facts)

    # load chroma from DB
    blog_content_db = Chroma(
        persist_directory=chroma_db_dir, embedding_function=embeddings
    )

    nearest_documents = blog_content_db.similarity_search(question, k=facts)

    # set_trace()
    for f in nearest_documents:
        if debug:
            ic(f.metadata["source"])

    facts = [Fact(source=f.metadata["source"], content=f.page_content) for f in nearest_documents]

    # explain what you want
    system_instructions = """
You are an expert at answering questions.
Use the passed in facts from Igor's blog to answer provided questions.

You give output in markdown
Before your answer, repeat the question as an H2 header

After you answer, return the list of sources and why they were relevant in order of relevance.
Be sure to include the % relvanace of each source
'_' is a valid part of the source file path, do not remove it
If there are multiple facts with the same source, combine them with indented bullet points
    """

    # give an example instead of trying to describe everything.
    system_example = """
E.g.

## the question the user asked here

### Answer

your answer here

### Sources

* source file path - Your reasoning on why it's  relevant (% relevance,  e.g. 20%)
    """

    system_prompt = system_instructions + system_example

    prompt = f"""
    ### Facts
{facts_to_prompt(facts)}
    ### Question
        {question}
    """
    resp = base_query(
        system_prompt=system_prompt, prompt_to_gpt=prompt, debug=debug, u4=u4
    )

    # We built the file_path from source markdown
    def fixup_markdown_path_to_url(src):
        markdown_to_url = build_markdown_to_url_map()
        for (md_file_path,url) in markdown_to_url.items():
            # url starts with a /
            url = url[1:]
            md_link = f"[{url}](https://idvork.in/{url})"
            src = src.replace(md_file_path, md_link)
        return src

    def fixup_ig66_path_to_url(src):
        for i in range(100*52):
            src =  src.replace(f"_ig66/{i}.md", f"[Family Journal {i}](https://idvork.in/ig66/{i})")
        return src

    out = fixup_markdown_path_to_url(resp)
    out = fixup_ig66_path_to_url(out)
    print(out)


# cache this so it's memoized
@lru_cache
def build_markdown_to_url_map():
    source_file_to_url = {}
    # read the json file From Github, slightly stale, but good enough
    backlinks_url = "https://raw.githubusercontent.com/idvorkin/idvorkin.github.io/master/back-links.json"
    d = requests.get(backlinks_url).json()
    url_infos = d["url_info"]
    # "url_info": {
        # "/40yo": {
            # "markdown_path": "_d/40-yo-programmer.md",
            # "doc_size": 14000
        # },
    # convert the url_infos into a source_file_to_url map
    source_file_to_url = {v["markdown_path"]: k for k, v in url_infos.items()}
    return source_file_to_url

@server.get("/remap/{source_file}")
def remap_to_url(source_file):
    return {"url": source_file_to_url(source_file)}
    # return {"url":"It works"}

@app.command()
@server.get("/remap/{source_file}")
def source_file_to_url(source_file):
    source_file_to_url = build_markdown_to_url_map()
    blog_base="https://idvork.in"
    return blog_base + source_file_to_url[source_file]

@app.command()
def ask2(
    question: Annotated[
        str, typer.Argument()
    ] = "What are the roles from Igor's Eulogy, answer in bullet form",
    facts: Annotated[int, typer.Option()] = 5,
    u4: bool = typer.Option(False),
    debug: bool = typer.Option(True),
):
    model = "gpt-4" if u4 else "gpt-3.5-turbo-16k"
    model, max_tokens = choose_model(u4)
    if debug:
        ic(model)
        ic(facts)
    # load chroma from DB
    blog_content_db = Chroma(
        persist_directory=chroma_db_dir, embedding_function=embeddings
    )
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.retrievers.multi_query import MultiQueryRetriever

    llm = ChatOpenAI(model_name=model)

    simple_retriever = blog_content_db.as_retriever(search_kwargs={"k": facts})
    smart_retriever = MultiQueryRetriever.from_llm(
        retriever=simple_retriever,
        llm=llm,
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=simple_retriever, verbose=True, return_source_documents=True
    )
    response = qa_chain({"query": question})
    print(response["result"])
    print("Source Documents")
    for doc in response["source_documents"]:
        ic(doc.metadata)


# @logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
