import os
from typing import List

from django.conf import settings
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document


def load_pdfs() -> List[Document]:
    chunks = []
    path = os.path.join(settings.BASE_DIR, "llm", "data", "sources")
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        is_pdf_file = os.path.isfile(filepath) and filename.endswith(".pdf")
        if is_pdf_file:
            loader = PyPDFLoader(filepath)
            chunks += loader.load_and_split()

    print(f"Loaded PDFs. Chunks: {len(chunks)}")

    return chunks
