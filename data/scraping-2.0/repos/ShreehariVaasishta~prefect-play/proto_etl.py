from pathlib import Path
from typing import List

import pypdf
import weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores.weaviate import Weaviate
from prefect import flow, task

WEAVIATE_URL = "http://127.0.0.1:8080"
wclient = weaviate.Client(url="http://localhost:7080")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)


# Define a Prefect task to read a chunk of PDF files
@task
def read_pdf(pdf_file: Path):
    with open(pdf_file, "rb") as f:
        pdf_reader = pypdf.PdfReader(f, strict=False)
        return [
            Document(
                page_content=page.extract_text(),
                metadata={"source": str(pdf_file.absolute()), "page": page_number},
            )
            for page_number, page in enumerate(pdf_reader.pages)
        ]


def dump_to_weaviate(documents):
    weaviatedb = Weaviate(
        client=wclient,
        by_text=False,
        index_name="Testing",
        text_key="test",
        embedding=embedding,
    )
    weaviatedb.add_documents(documents)


@task
def dump_embeddings(pdf_file, cnt):
    print(cnt, "Finished <<<<<")
    return dump_to_weaviate(pdf_file)


@flow(log_prints=True)
def run_flow():
    directory = (
        "/home/sln/VFS/master-gdc-gdcdatasets-2020445568-2020445568/lcwa_gov_pdf_data/data"  # Directory of PDFs
    )

    pdf_files: List[Path] = list(Path(directory).glob("**/*.pdf"))

    for i, pdf_chunk in enumerate(pdf_files):
        chunk_task = read_pdf.submit(pdf_chunk)
        dump_embeddings.submit(chunk_task.result(), i)
        print(f"Processed {i} of chunk  {len(pdf_files)}")
        # if i > 5:
        #     break


run_flow()
