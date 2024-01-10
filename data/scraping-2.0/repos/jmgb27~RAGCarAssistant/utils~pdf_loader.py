from langchain.document_loaders import PDFMinerLoader

def load_pdf(pdf_path, text_splitter):
    loader = PDFMinerLoader(pdf_path)
    return loader.load_and_split(text_splitter=text_splitter)
