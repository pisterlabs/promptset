from langchain.docstore.document import Document

from loaders import loaders


def get_url_documents(url: str) -> list[Document]:
    loader = loaders.locate(url)
    if loader is not None:
        docs = loader.load()
        if docs:
            return docs

    raise RuntimeError(f"Cannot extract documents from {url}")
