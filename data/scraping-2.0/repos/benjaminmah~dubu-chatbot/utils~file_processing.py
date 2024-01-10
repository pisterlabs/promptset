import fitz
from typing import List
from langchain.docstore.document import Document


def get_documents_from_file_stream(stream) -> List[Document]:
    with fitz.open(stream=stream) as doc:
        return [
            Document(
                page_content=page.get_text().encode("utf-8"),
                metadata=dict(
                    {
                        "id": page.number + 1,
                        "page_number": page.number + 1,
                        "total_pages": len(doc),
                    }
                ),
            )
            for page in doc
        ]

def get_documents_from_file(input_file):
    with open(input_file, "rb") as fh:
        ba = bytearray(fh.read())
        return get_documents_from_file_stream(ba)