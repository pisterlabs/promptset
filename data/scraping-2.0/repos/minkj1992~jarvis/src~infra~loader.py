"""Load Documenent from outside
for example urls, pdf, excel, .doc file, youtube ...
"""

import mimetypes
from enum import Enum
from typing import IO, List

from fastapi.concurrency import run_in_threadpool
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.unstructured import UnstructuredFileIOLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.file_utils.filetype import FileType
from unstructured.partition.auto import partition

from app.exceptions import FileEmptyContentsException, InvalidFileExtException
from app.logger import get_logger
from infra.config import get_config

logger = get_logger(__name__)
cfg = get_config()

class UnstructuredFileStrategy(str, Enum):
    FAST = "fast"
    HI_RES = "hi_res"
    OCR_ONLY = "ocr_only"

class JarvisFileLoader:
    """Jarvis file loader class for multi-part files.

    `Unstructured` library is needed for OCR.
    You can check `unstructured` support types from `FileType`

    from unstructured.file_utils.filetype import FileType    
    ".pdf"
    ".docx"
    ".jpg"
    ".jpeg"
    ".txt"
    ".text"
    ".eml"
    ".xml"
    ".html"
    ".md"
    ".xlsx"
    ".pptx"
    ".png"
    ".doc"
    ".zip"
    ".xls"
    ".ppt"
    ".rtf"
    ".json"
    ".epub"
    ".msg"
    ".odt"
    """


    @staticmethod
    async def load(file_name: str, file: IO) -> List[Document]:
        """load document from pdf bytes data"""
        mime_type = mimetypes.guess_type(file_name)[0]

        if mime_type == "text/csv":
            ...
        else:
            try:
                if mime_type in (None, ''):
                    elements = partition(file=file, file_filename=file_name, strategy=UnstructuredFileStrategy.FAST)
                else:
                    elements = partition(file=file, file_filename=file_name, content_type=mime_type,strategy=UnstructuredFileStrategy.FAST)
                if elements == []:
                    raise FileEmptyContentsException(file_name=file_name)
                text = "\n\n".join([str(el) for el in elements])
                metadata = {"source": file_name}
            except ValueError:
                raise InvalidFileExtException(file_name=file_name, mime_type=mime_type)
            except Exception as e:
                await logger.exception(e)


        return [Document(page_content=text, metadata=metadata)]
        

    @staticmethod
    async def load_and_split(file_name: str, file: IO) -> List[Document]:
        """split docs into chunks."""
        _text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=20, separators=["\n\n", "\n", " ", ""])

        docs = await JarvisFileLoader.load(file_name, file)
        return _text_splitter.split_documents(docs)



# class JarvisCSVLoader:
#     """Loads a CSV file into a list of documents.

#     Each document represents one row of the CSV file. Every row is converted into a
#     key/value pair and outputted to a new line in the document's page_content.

#     The source for each document loaded from csv is set to the value of the
#     `file_path` argument for all doucments by default.
#     You can override this by setting the `source_column` argument to the
#     name of a column in the CSV file.
#     The source of each document will then be set to the value of the column
#     with the name specified in `source_column`.

#     Output Example:
#         .. code-block:: txt

#             column1: value1
#             column2: value2
#             column3: value3
#     """

#     def __init__(
#         self,
#         file_path: str,
#         source_column: Optional[str] = None,
#         csv_args: Optional[Dict] = None,
#         encoding: Optional[str] = None,
#     ):
#         self.file_path = file_path
#         self.source_column = source_column
#         self.encoding = encoding
#         if csv_args is None:
#             self.csv_args = {
#                 "delimiter": ",",
#                 "quotechar": '"',
#             }
#         else:
#             self.csv_args = csv_args

#     def load(self) -> List[Document]:
#         docs = []

#         with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
#             csv = DictReader(csvfile, **self.csv_args)  # type: ignore
#             for i, row in enumerate(csv):
#                 content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items())
#                 if self.source_column is not None:
#                     source = row[self.source_column]
#                 else:
#                     source = self.file_path
#                 metadata = {"source": source, "row": i}
#                 doc = Document(page_content=content, metadata=metadata)
#                 docs.append(doc)

#         return docs


async def from_file(file_name: str, file: IO) -> List[Document]:
    return await JarvisFileLoader.load_and_split(file_name, file)


def from_pdf():
    ...


def from_doc():
    ...


def from_youtube():
    ...


def from_url():
    ...

def from_urls():
    ...


