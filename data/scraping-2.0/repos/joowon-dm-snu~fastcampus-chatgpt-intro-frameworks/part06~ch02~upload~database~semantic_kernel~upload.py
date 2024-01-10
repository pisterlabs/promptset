import json
import os
from uuid import uuid4

import markdown
import nbformat
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore
from semantic_kernel.text.text_chunker import (
    split_markdown_paragraph,
    split_plaintext_paragraph,
)

load_dotenv()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CUR_DIR), "dataset")

SK_CODE_DIR = os.path.join(DATA_DIR, "semantic-kernel", "python")
SK_SAMPLE_DIR = os.path.join(
    DATA_DIR, "semantic-kernel", "samples", "notebooks", "python"
)
SK_DOC_DIR = os.path.join(DATA_DIR, "semantic-kernel-docs", "semantic-kernel")

CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "chroma-persist")
CHROMA_COLLECTION_NAME = "fastcampus-bot"


kernel = Kernel()
kernel.add_text_embedding_generation_service(
    "ada",
    OpenAITextEmbedding(
        "text-embedding-ada-002",
        os.getenv("OPENAI_API_KEY"),
    ),
)
kernel.register_memory_store(
    memory_store=ChromaMemoryStore(persist_directory=CHROMA_PERSIST_DIR)
)


def read_file(file_path):
    with open(file_path, "r") as f:
        if file_path.endswith(".ipynb"):
            nb = nbformat.read(f, as_version=4)
            contents = ""
            for cell in nb["cells"]:
                if cell["cell_type"] in ["code", "markdown"]:
                    contents += cell["source"] + "\n"
                else:
                    raise ValueError(f"Unknown cell type: {cell['cell_type']}")
        else:
            contents = f.read()

    if file_path.endswith(".ipynb") or file_path.endswith(".md"):
        contents = markdown.markdown(contents)
        soup = BeautifulSoup(contents, "html.parser")
        contents = soup.get_text()

    return contents


async def upload_embedding_from_file(file_path):
    contents = read_file(file_path)

    if file_path.endswith(".ipynb") or file_path.endswith(".md"):
        chunks = split_markdown_paragraph([contents], max_tokens=500)
    else:
        chunks = split_plaintext_paragraph([contents], max_tokens=500)

    for chunk_id, chunk in enumerate(chunks):
        await kernel.memory.save_information_async(
            collection=CHROMA_COLLECTION_NAME,
            text=chunk,
            id=str(uuid4()),
            description=os.path.relpath(file_path, DATA_DIR),
            additional_metadata=json.dumps({"chunk_id": chunk_id}),
        )


async def upload_embeddings_from_dir(dir_path):
    failed_upload_files = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py") or file.endswith(".md") or file.endswith(".ipynb"):
                file_path = os.path.join(root, file)
                await upload_embedding_from_file(file_path)
                try:
                    await upload_embedding_from_file(file_path)
                    print("SUCCESS: ", file_path)
                except Exception:
                    print("FAILED: ", file_path)
                    failed_upload_files.append(file_path)


if __name__ == "__main__":
    import asyncio

    asyncio.run(upload_embeddings_from_dir(SK_CODE_DIR))
    asyncio.run(upload_embeddings_from_dir(SK_SAMPLE_DIR))
    asyncio.run(upload_embeddings_from_dir(SK_DOC_DIR))
