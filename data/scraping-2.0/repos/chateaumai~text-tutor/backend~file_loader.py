from langchain.document_loaders import PyMuPDFLoader
from .PDF_Processing import crop

#getting texts and title
def load_file(file_name):
    path, title = crop(file_name)
    loader = PyMuPDFLoader(path)
    data = loader.load()
    page_contents = [doc.page_content for doc in data]
    return page_contents, title