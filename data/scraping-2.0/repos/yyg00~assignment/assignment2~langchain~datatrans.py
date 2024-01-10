from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from config import parsers


class DataLoader(object):
    def __init__(self):
        self.args = parsers()
        self.zh_file_path = self.args.src
        self.en_file_path = self.args.tgt

    def get_data(self):
        zh_loader = PyPDFLoader(self.zh_file_path)
        en_loader = PyPDFLoader(self.en_file_path)

        zh_data = zh_loader.load()
        en_data = en_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        zh_texts = text_splitter.split_documents(zh_data)
        en_texts = text_splitter.split_documents(en_data)

        return zh_texts, en_texts
