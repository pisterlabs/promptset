from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document


def create_fixed_size_chunks(
    text: list[str],
    separator: str = "\n\n",
    chunk_size: int = 512,
    chunk_overlap: int = 20,
) -> list[Document]:
    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return text_splitter.create_documents(text)


loader = TextLoader("../data/test.txt")

text_splitter = CharacterTextSplitter(separator=":", chunk_size=20, chunk_overlap=0)

text_splitter = NLTKTextSplitter()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=512,
    chunk_overlap=20,
)

docs = loader.load_and_split(text_splitter=text_splitter)

for doc in docs:
    print(doc.page_content)
    print("\n")
