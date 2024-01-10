from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


class Preprocessing:
    def __init__(self, filepath):
        self.filepath = filepath

    @staticmethod
    def split_pdf(filepath):
        reader = PdfReader(filepath)

        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        text_splitter = RecursiveCharacterTextSplitter(
            # separator = "\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            )

        texts = text_splitter.split_text(raw_text)
        return texts

    def process_files(self):
        all_texts = []
        for file in self.filepath:
            all_texts.append(self.split_pdf(file))
        return all_texts
