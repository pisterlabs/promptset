import json
import os
import re

from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from core.const import DataSource, SUCCESS_PATH
from core.llm import get_or_create_chain
from rxconfig import config

SUMMARY: dict = {}


def preprocessing(data_source: str | DataSource):
    if isinstance(data_source, str):
        data_source = DataSource(data_source)

    full_texts = data_source.load()

    # 1. first_line -> subject
    first_line, *lines = full_texts
    subject = first_line.replace(":", "").strip()
    lines = [f"# {subject}"] + lines

    # 2. '-' -> '- '로 변환
    new_lines = []
    for line in lines:
        if line.startswith("-"):
            line = line[1:].strip()
            line = f"- {line}"
        new_lines.append(line.strip())
    lines = "\n".join(new_lines).split("\n")

    # 3. table to list
    new_lines = []
    table_contents = columns = None
    for i, line in enumerate(lines):

        # parse table
        if "|" in line:
            if table_contents is None:
                # parse columns
                table_contents = []
                columns = [column.strip() for column in line.split("|")]
                continue

            row = line.split("|")[:len(columns)]
            parsed_row = {
                column: value.strip()
                for column, value in zip(columns, row)
            }
            parsed_line = "\n".join([f"\t{column}: {value}" for column, value in parsed_row.items()])
            new_lines.append(f"- " + parsed_line)
            continue

        if isinstance(table_contents, list):
            if "|" not in line or (i == len(lines) - 1):
                table_contents = None
                continue
        new_lines.append(line.strip())
    lines = "\n".join(new_lines).split("\n")

    # 4. '#' -> ## h2로 변환
    new_lines = []
    for line in lines:
        if line.startswith("#"):
            line = line.replace("#", "").strip()
            line = f"\n## {line}\n"
            new_lines.append(line)
            continue
        new_lines.append(line)
    lines = "\n".join(new_lines).split("\n")

    # 5. '1.' -> '### 1. '로 변환
    new_lines = []
    for line in lines:
        if re.match(r"^\d+\.", line):
            # 정규식으로 '1.'을 '1. '로 변환
            line = re.sub(r"^\d+\.", r"\g<0> ", line)
            new_lines.append(f"### {line}\n")
            continue
        new_lines.append(line)

    # concat
    full_text = "\n".join(new_lines)

    # save preprocessed data
    data_source.dump(full_text)

    # load and split
    loader = UnstructuredMarkdownLoader(
        data_source.dest_path,
        mode="elements",
        strategy="fast",
    )
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    docs = loader.load_and_split(splitter)

    # add metadata & pop unnecessary metadata
    for doc in docs:
        doc.metadata["data_source"] = f"{data_source}"
        doc.metadata.pop("file_directory", None)
        doc.metadata.pop("source", None)
        doc.metadata.pop("filetype", None)
        doc.metadata.pop("last_modified", None)
        doc.metadata.pop("page_number", None)

    # upload
    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
        collection_name=config.CHROMA_COLLECTION_NAME,
    )

    return full_text, len(docs)


def load_data_and_upload_chroma():
    global SUMMARY

    if os.path.exists(SUCCESS_PATH):
        with open(SUCCESS_PATH, "r+", encoding="utf-8") as file:
            SUMMARY = json.load(file)
        return

    summarize_chain = get_or_create_chain("summarize")

    # do preprocess for all data sources
    for data_source in DataSource:
        if isinstance(data_source, str):
            data_source = DataSource(data_source)

        # preprocessing
        full_text, count = preprocessing(data_source)

        # get keywords for each data source
        summarized_text = summarize_chain.run({"text": full_text})

        SUMMARY[f"{data_source}"] = summarized_text.replace("\n", "")
    print(SUMMARY)

    with open(SUCCESS_PATH, "w+", encoding="utf-8") as file:
        json.dump(SUMMARY, file, ensure_ascii=False, indent=4)


def get_summary() -> dict:
    global SUMMARY
    if not SUMMARY:
        load_data_and_upload_chroma()
    return SUMMARY
