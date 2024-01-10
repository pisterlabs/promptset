from enum import Enum
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import CSVLoader


class FileType(Enum):
    CSV = 'csv'
    DOCX = 'docx'
    # MP4 = 'mp4'
    PDF = 'pdf'
    PPTX = 'pptx'


file_handler_map = {
    # FileType.MP4: handle_mp4,
    FileType.CSV: CSVLoader,
    FileType.DOCX: Docx2txtLoader,
    FileType.PDF: UnstructuredPDFLoader,
    FileType.PPTX: UnstructuredPowerPointLoader,
}
