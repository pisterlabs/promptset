import re
from io import BytesIO

from langchain.docstore.document import Document
from pypdf import PdfReader


def parse_pdf(file: str | BytesIO) -> list[Document]:
    """Parse pdf file and return list of document pages.
    :param file: str or BytesIO object of pdf file
    :returns: List of Document objects
    """
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(Document(page_content=text))
    return output
