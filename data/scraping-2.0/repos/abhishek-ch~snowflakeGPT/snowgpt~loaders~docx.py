from .common import process_file
from langchain.document_loaders import Docx2txtLoader
from sqlite3 import Connection


def process_docx(file, stats_db):
    return process_file(file, Docx2txtLoader, ".docx", stats_db=stats_db)
