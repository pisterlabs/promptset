from uuid import uuid4

from langchain.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import LatexTextSplitter

def get_academy_docs() -> list[Document]:
    loader = DirectoryLoader(
        './docs',  # docs-small
        use_multithreading=True,
        show_progress=True,
        loader_cls=TextLoader,
    )
    docs = loader.load()
    # splitter = MarkdownTextSplitter(
    #     chunk_size=2500,
    #     chunk_overlap=200
    # )
    splitter = LatexTextSplitter(
        chunk_size=2500,
        chunk_overlap=200
    )
    splits_documents = splitter.split_documents(docs)
    print("load document chunks: ", len(docs))
    for doc in splits_documents:
        source: str = doc.metadata['source']
        doc.metadata['doc_id'] = str(uuid4())
        doc.metadata['doc_name'] = source.replace("docs/", "")
    return splits_documents

def get_demo_docs() -> list[Document]:
    loader = DirectoryLoader(
        './demo',
        glob=f"**/*.pdf",
        show_progress=True,
        loader_cls=PyMuPDFLoader,
    )
    docs = loader.load()
    # splitter = MarkdownTextSplitter(
    #     chunk_size=2500,
    #     chunk_overlap=200
    # )
    splitter = LatexTextSplitter(
        chunk_size=2500,
        chunk_overlap=200
    )
    splits_documents = splitter.split_documents(docs)
    print("load document chunks: ", len(docs))
    for doc in splits_documents:
        source: str = doc.metadata['source']
        doc.metadata['doc_id'] = str(uuid4())
        doc.metadata['doc_name'] = source.replace("docs/", "")
    return splits_documents