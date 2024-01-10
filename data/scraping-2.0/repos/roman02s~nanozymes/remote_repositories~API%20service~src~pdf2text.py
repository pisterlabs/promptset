import os


from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from src.logger import Logger

def process_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if len(line.strip()) > 2]
    text = "\n".join(lines).strip()
    if len(text) < 10:
        return None
    return text


class PDF2text:
    LOADER_MAPPING = {
        ".pdf": (PDFMinerLoader, {})
    }
    def __init__(self, file_paths, chunk_size=100, chunk_overlap=0):
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def __load_single_document(self, file_path: str) -> Document:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext not in self.LOADER_MAPPING:
            for ext in self.LOADER_MAPPING:
                Logger.info(file_path + ext)
                if os.path.exists(file_path + ext):
                    file_path += ext
                    break
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        loader_class, loader_args = self.LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    def build_index(self):
        documents = [self.__load_single_document(path) for path in self.file_paths]
        # Logger.info(documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        documents = text_splitter.split_documents(documents)
        # Logger.info(documents)
        self.fixed_documents = []
        for doc in documents:
            doc.page_content = process_text(doc.page_content)
            if not doc.page_content:
                continue
            self.fixed_documents.append(doc)

        Logger.info(f"Загружено {len(self.fixed_documents)} фрагментов! Можно задавать вопросы.")
        return self.fixed_documents

# Пример использования PDF2text
if __name__ == "__main__":
    file_paths = ["data/C4RA15675G.pdf"]
    chunk_size = 200
    chunk_overlap = 10

    pdf2text = PDF2text(file_paths, chunk_size, chunk_overlap)
    fixed_documents = pdf2text.build_index()
