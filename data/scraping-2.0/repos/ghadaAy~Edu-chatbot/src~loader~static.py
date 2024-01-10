from langchain.document_loaders import (
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    UnstructuredImageLoader
)

encoding = {"encoding": "utf8"}
LOADER_MAPPING: dict = {
    ".html": (UnstructuredHTMLLoader, encoding),
    ".md": (UnstructuredMarkdownLoader, encoding),
    ".pdf": (PyPDFLoader, {}),
    ".txt": (TextLoader, encoding),
    ".png":(UnstructuredImageLoader, {}),
    ".jpg":(UnstructuredImageLoader, {}),
    ".jpeg":(UnstructuredImageLoader, {}),
}
