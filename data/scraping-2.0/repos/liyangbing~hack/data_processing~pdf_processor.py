# from .data_processor import DataProcessor
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import NLTKTextSplitter


class PdfProcessor:

    def __init__(self, ck_size, ck_overlap_window):
        self.ck_size = ck_size
        self.ck_overlap_window = ck_overlap_window

    def load_data(self, fpath):
        loader = PyMuPDFLoader(fpath)
        data = loader.load()
        return data

    def split_data(self, document_list):
        text_splitter = NLTKTextSplitter(
            chunk_size=self.ck_size,
            chunk_overlap=self.ck_overlap_window,
        )
        text_data = text_splitter.split_documents(document_list)
        return text_data

    def display_info(self, data):
        print(f'There are {len(data)} documents in your data. ')
        print(
            f'There are {len(data[0].page_content)} characters in your first document. ')


if __name__ == "__main__":
    test_fpath = '../dataset'
    pdf_data = PdfProcessor(250, 10)
    raw_data = pdf_data.load_data(test_fpath)
    data_list = pdf_data.split_data(raw_data)
    pdf_data.display_info(data_list)
    # res_data = pdf_data.data_load(test_fpath)
    # res_data = pdf_data.create_docs(test_fpath)
    print(data_list[0])
