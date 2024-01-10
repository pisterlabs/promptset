from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter

from ragger_duck.scraping import UserGuideDocExtractor
from ragger_duck.scraping._shared import _chunk_document

TEST_DATA_PATH = Path(__file__).parent / "data" / "user_guide_doc"


def test_chunk_document():
    """Check the behavior of the `_chunk_document` function."""
    chunk_size = 100
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
    )
    extractor = UserGuideDocExtractor(chunk_size=None)
    document_unchunked = extractor.fit_transform(TEST_DATA_PATH)[0]
    chunks = _chunk_document(text_splitter, document_unchunked)
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert len(chunk["text"]) <= chunk_size
        assert chunk["source"] in document_unchunked["source"]
