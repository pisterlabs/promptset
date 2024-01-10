from langchain.document_loaders import UnstructuredPDFLoader


def pdf(filename):
    loader = UnstructuredPDFLoader(filename)
    return loader.load()
