import os
from uuid import uuid4

import markdown
import nbformat
import semantic_kernel as sk
from bs4 import BeautifulSoup
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, "dataset")

SK_CODE_DIR = os.path.join(DATA_DIR, "semantic-kernel", "python")
SK_SAMPLE_DIR = os.path.join(DATA_DIR, "semantic-kernel", "samples", "python")
SK_DOC_DIR = os.path.join(DATA_DIR, "semantic-kernel-docs", "semantic-kernel")

CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "chroma-persist")
CHROMA_COLLECTION_NAME = "fastcampus-bot"


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


async def upload_embeddings_from_file(file_path):
    contents = read_file(file_path)

    await kernel.memory.save_information_async(
        collection=CHROMA_COLLECTION_NAME,
        text=contents,
        id=str(uuid4()),
        description=os.path.relpath(file_path, DATA_DIR),
    )


async def upload_embeddings_from_dir(dir):
    failed_upload_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in [".py", ".md", ".ipynb"]:
                file_path = os.path.join(root, file)

                try:
                    await upload_embeddings_from_file(file_path)
                    print("SUCCESS:", file_path)
                except Exception:
                    print("FAILED:", file_path)
                    failed_upload_files.append(file_path)

    return failed_upload_files


if __name__ == "__main__":
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()

    kernel = sk.Kernel()
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
    failed_doc_files = asyncio.run(upload_embeddings_from_dir(SK_DOC_DIR))
    failed_codebase_files = asyncio.run(upload_embeddings_from_dir(SK_SAMPLE_DIR))
    failed_sample_files = asyncio.run(upload_embeddings_from_dir(SK_CODE_DIR))
    # print(failed_doc_files)
    # print(failed_codebase_files)
