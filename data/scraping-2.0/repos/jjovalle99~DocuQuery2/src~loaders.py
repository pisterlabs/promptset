from io import BytesIO
from typing import List

from chainlit.types import AskFileResponse
from langchain.docstore.document import Document
from pypdf import PdfReader


def get_docs(files: List[AskFileResponse], splitter) -> List[str]:
    docs = []
    for file in files:
        reader = PdfReader(BytesIO(file.content))
        doc = [
            Document(
                page_content=page.extract_text(),
                metadata={"source": file.path, "page": page.page_number},
            )
            for page in reader.pages
        ]
        docs.append(doc)
    splitted_docs = [splitter.split_documents(doc) for doc in docs]
    for doc in splitted_docs:
        for i, chunk in enumerate(doc, start=1):
            chunk.metadata["chunk"] = i
    unnested_splitted_docs = [chunk for doc in splitted_docs for chunk in doc]
    return unnested_splitted_docs
